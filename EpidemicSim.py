# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 11:38:03 2020

@author: Patrick Raphael
"""

import numpy as np
import time
import datetime
from enum import IntEnum
import sys

import ctypes
import numpy.ctypeslib as np_ctypes
import os

# node types
class Node(IntEnum):
    home = 0         # residence
    private_work = 1 # offices/factories/etc.- daytime working hours, not open to public
    retail = 2       # retail daytime-evening working hours
    school = 3       # schools- daytime working hours
    hospital = 4     # hospitals

# state of the infection for each individual
       
class DiseaseState(IntEnum):
    not_infected = 0
    infected_not_contagious = 1
    infected_contagious = 2
    minor_symptoms = 3
    sick = 4
    hospitalized = 5
    recovery_contagious = 6
    recovery_not_contagious = 7
    deceased = 8
    
    
# native helper library improves speed 
useHelperLibrary = True

# condition for shutting down schools
# number of sick, hospitalized or dead individuals
schoolShutdownCount = 50
schoolShutdownCondition = DiseaseState.sick
# condition for shutting down non-essential business 
# number of sick, hospitalized or dead individuals
nonEssentialBusinessShutdownCount = 200
nonEssentialBusinessShutdownCondition = DiseaseState.sick

# how many days to run the simulation
timeSimDays = 90
# of times to run the simulation (for averaging) 
numRuns = 1  
        
# US population distirbution (rough) in millions, in increments of 10 years
# 0-9, 10-19, 20-29 etc.
age_us_millions = np.array([40, 40, 46, 43, 39, 42, 35, 20, 10, 4])
pop_us_total = np.sum(age_us_millions)
age_distr = age_us_millions / pop_us_total
age_distr_cs = np.cumsum(age_distr)

# distrubtion of household size from 1 to 7
household_distr = np.array([0.28, 0.34, 0.15, 0.13, 0.06, 0.022, 0.011, 0.007])
household_distr_cs = np.cumsum(household_distr)
household_size_max = len(household_distr)

child_pop_distr = np.sum(household_distr[0:1])
adult_pop_distr = np.sum(household_distr[2:6])
senior_pop_distr = np.sum(household_distr[7:9])

# sim params
population = 500000
node_size_max = 1000   # of people that can occupy nonhome node in a given time

# private work sizes for small medium large 
private_work_sizes_avg = 50
private_work_sizes_std = 15
 
# sizes for retail small medium large 
retail_sizes_avg = 25
retail_sizes_std = 5

school_size_avg = 500
school_size_std = 200
school_staff_ratio = 0.05
hospital_size_avg = 1000
hospital_staff_ratio = 0.05

# percentage of retailers and work which are considered essential
esssential_retail_fraction = 0.1   
esssential_private_fraction = 0.1

# max number of different nodes for visitation
visit_node_max = 20   
visits_per_schedule = 4  # how many times will visit 

# time phases over 24 hour period
# phase 0: late-night early morning 
# passe 1: mornining 8 AM to 12 PM
# phase 2: afternoon 12 PM - 4 PM
# phase 3: early evening  4 PM - 8 PM
# phase 4: late evening 8 PM - 12 PM
time_phases_per_day = 5  

# repeating days to schedule people
schedule_days = 7

# disease parameters
contagious_time = 2*time_phases_per_day   # time from onset before disease is contagious after being infected
incubation_time = 5*time_phases_per_day   # time from onset before symptoms appear
hospitalization_time = 10*time_phases_per_day           # time from onset before hospitalization
recovery_time_minor_symptoms = 15*time_phases_per_day   # time from onset to recovery if only minor symptoms
recovery_time_sick = 18*time_phases_per_day             # time from onset to recovery if sick (no hospitalization)
recovery_time_hospitalization = 24*time_phases_per_day  # time from onset to recovery for hospitalized patients
recovery_time_contagious = 5*time_phases_per_day        # time it take for recovery to advance from contagious to non-contagious 
death_time = 16*time_phases_per_day                     # time from onset until death

initial_infections = 5
infection_chance_per_time = 0.01
mortality_vs_age = np.array([0.001, 0.001, 0.001, 0.001, 0.002, 0.005, 0.01, 0.07, 0.15, 0.25])
# fraction of infected individual who only exhibit minor symptoms
minor_symptoms_fraction = 0.5  
# fraction of sick patients who need to be hospitalized
hospitalization_fraction = 0.2
mortality_vs_age_hospitalization = mortality_vs_age / hospitalization_fraction
  
# maximum number of pople that a single individual can infect in a single time unit
infection_max_people_per_time = 50   

senior_population = int(round(population * (1 - age_distr_cs[7])))
child_population = int(round(population * age_distr_cs[1]))
adult_population = population - senior_population - child_population 

num_private_work_nodes = int(round(adult_population / 100)) + 1
num_retail_nodes = int(round(population / 100)) + 1
num_school_nodes = int(round(child_population / school_size_avg)) + 1 
num_hospital_nodes = int(round(population / 10000)) + 1
num_nonhome_nodes = num_private_work_nodes + num_school_nodes + num_retail_nodes + num_hospital_nodes

avg_household_sz = np.sum(household_distr * np.arange(1, household_size_max+1)) 
num_home_nodes = int(round(1.25*population/avg_household_sz)) + 5 # allow some excess housing
num_nodes = num_home_nodes + num_nonhome_nodes


class epi_sim:
    
    def initHelperLibrary(self):
        pth = 'C:\\Users\\praph\\OneDrive\\Documents\\Python Scripts'
        dll_name = 'EpidemicSimHelper'
        dll_path = os.path.join(pth, dll_name)
        self.helperlib = ctypes.CDLL(dll_path)
        data_arr_t = np_ctypes.ndpointer(ctypes.c_int32, flags="C_CONTIGUOUS")
        uint32_arr_t = np_ctypes.ndpointer(ctypes.c_uint32, flags="C_CONTIGUOUS")
        self.helperlib.arrayTest.argtypes = [data_arr_t, ctypes.c_int32]        
        char_arr_t = np_ctypes.ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS")
        float_arr_t = np_ctypes.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")
        bool_arr_t = np_ctypes.ndpointer(ctypes.c_bool, flags="C_CONTIGUOUS")
        self.helperlib.SetArrays.argtypes = [data_arr_t, data_arr_t, data_arr_t, data_arr_t,
                                             data_arr_t, data_arr_t, char_arr_t, char_arr_t, 
                                             data_arr_t, uint32_arr_t, char_arr_t, data_arr_t,
                                             data_arr_t]
        
        ct_int32 = ctypes.c_int32
        ct_float = ctypes.c_float

        self.helperlib.SetDiseaseParams.argtypes = [ct_int32, ct_int32, ct_float, ct_int32, 
                                                    ct_int32, ct_int32, ct_int32, ct_float, 
                                                    ct_int32, float_arr_t]
        self.helperlib.movePopulation.argtypes = [ct_int32, ct_int32, ct_int32, bool_arr_t]
        self.helperlib.spreadInfection.argtypes = [ct_int32, ct_float, ct_int32]
        self.helperlib.scheduleVisits.argtypes = [data_arr_t, data_arr_t, ct_int32, ct_int32]
        self.helperlib.assignVisitNodes.argtypes = [data_arr_t, data_arr_t, ct_int32, ct_int32, ct_int32, ct_int32]
        self.helperlib.SetParams(population, num_nodes, node_size_max, schedule_days, time_phases_per_day)
        self.helperlib.SetNodeParams(num_home_nodes, num_hospital_nodes, num_retail_nodes, 
                                     self.num_essential_retail, self.hospital_node_idx_begin,
                                     self.retail_node_idx_begin)
        
        mortality_vs_age_hospitalization = mortality_vs_age / hospitalization_fraction
        mortality_vs_age_hospitalization = np.require(mortality_vs_age_hospitalization, np.float32, ['C', 'W'])
        self.helperlib.SetDiseaseParams(contagious_time, incubation_time, minor_symptoms_fraction,
                                        recovery_time_minor_symptoms, recovery_time_contagious, 
                                        recovery_time_hospitalization, hospitalization_time,
                                        hospitalization_fraction, death_time,
                                        mortality_vs_age_hospitalization)
        
        self.all_nodes = np.require(self.all_nodes, np.int32, ['C', 'W'])
        self.home_node_prsn = np.require(self.home_node_prsn, np.int32, ['C', 'W'])
        self.work_school_node_prsn = np.require(self.work_school_node_prsn, np.int32, ['C', 'W'])
        self.current_node_prsn = np.require(self.current_node_prsn, np.int32, ['C', 'W'])
        self.current_node_slot_prsn = np.require(self.current_node_slot_prsn, np.int32, ['C', 'W'])
        self.node_schedule_prsn_temp = np.require(self.node_schedule_prsn_temp, np.int32, ['C', 'W'])
        self.infection_state = np.require(self.infection_state, np.uint8, ['C', 'W'])
        self.current_prsn_loc = np.require(self.current_prsn_loc, np.uint8, ['C', 'W'])
        self.current_node_size = np.require(self.current_node_size, np.int32, ['C', 'W'])
        self.max_node_size = np.require(self.max_node_size, np.uint32, ['C', 'W'])
        self.age_prsn = np.require(self.age_prsn, np.uint8, ['C', 'W'])
        self.infection_start_time = np.require(self.infection_start_time, np.int32, ['C', 'W']) 
        self.infection_recovery_start_time = np.require(self.infection_recovery_start_time, np.int32, ['C', 'W'])
        self.isNodeShutdown = np.require(self.infection_recovery_start_time, np.bool, ['C', 'W'])
        
        # set the array pointers in DLL 
        self.helperlib.SetArrays(self.all_nodes, self.home_node_prsn, self.work_school_node_prsn, 
                                 self.current_node_prsn, self.current_node_slot_prsn, 
                                 self.node_schedule_prsn_temp, self.infection_state,
                                 self.current_prsn_loc, self.current_node_size, self.max_node_size, 
                                 self.age_prsn, self.infection_start_time, self.infection_recovery_start_time)

        
    # resets simulation to intial state
    # all individuals st home
    def reset(self):
        # current state of nodes -1 == no occupy, >0 means occupied
        self.all_nodes[:, :] = -1
        
        self.current_node_size[:] = 0
        self.current_prsn_loc[:] = 0
        self.current_node_slot_prsn[:] = -1
        self.current_node_prsn[:] = -1
        
        self.infection_state[:] = DiseaseState.not_infected
        # start time of the infection in time units
        self.infection_start_time[:] = -1
        
        self.infection_recovery_start_time[:] = -1
        # next patient ot be hospitalized will go to this hospital
        self.next_hospital_idx = self.hospital_node_idx_begin
        
        self.isNodeShutdown[:] = False
        
        # assigna all people to home
        for p in range(0, population):
            home_nd_idx = self.home_node_prsn[p]
            sz = self.current_node_size[home_nd_idx]
            if(home_nd_idx < 0 or home_nd_idx >= num_home_nodes):
                print("ERROR bad home_nd_idx %d  p= %d age= %d" % (home_nd_idx, p, self.age_prsn[p]))
                continue
            self.all_nodes[home_nd_idx][sz] = p
            
            # person is initially at home
            self.current_node_prsn[p] = home_nd_idx
            self.current_node_slot_prsn[p] = sz
            self.current_node_size[home_nd_idx] += 1     

        # infect a few individuals at the start
        self.infection_state[self.initial_infection_idx] = DiseaseState.infected_contagious
        self.infection_start_time[self.initial_infection_idx] = -contagious_time;
        
    def assignHomes(self):
        # assign individuals to homes    
        
        # construct random permutation of different populations
        # we need to offset adult and senikor permutation indices because 
        # we constructed age_prsn    
        child_perm = np.random.permutation(child_population)
        adult_perm = np.random.permutation(adult_population) + child_population
        senior_perm = np.random.permutation(senior_population) + child_population + adult_population

        adult_to_senior_frac = adult_pop_distr/(adult_pop_distr + senior_pop_distr)
        
        child_idx = 0
        adult_idx = 0
        senior_idx = 0
        
        # probably that a home has both adults and seniors
        adult_senior_mix_prob = 0.05
        home_idx = 0
        
        # keep track of how many adults, senoirs and children in home node
        self.home_node_num_children = np.zeros(num_home_nodes, dtype=np.uint8)
        self.home_node_num_adults = np.zeros(num_home_nodes, dtype=np.uint8)
        self.home_node_num_seniors = np.zeros(num_home_nodes, dtype=np.uint8)
        
        while(home_idx < num_home_nodes):
            node_size = self.typical_node_size[home_idx]            
            
            n_adults = 0
            n_children = 0
            n_seniors = 0
            
            # first indivdual, must be adult or senior
            rnd = np.random.ranf()
            if(rnd < adult_to_senior_frac):
                if(adult_idx < adult_population):
                    n_adults = 1
                else:
                    if(senior_idx < senior_population):
                        n_seniors = 1
            else:
                if(senior_idx < senior_population):
                    n_seniors = 1
                else:
                    if(adult_idx < adult_population):
                        n_adults = 1
                
            # ideally this shouldn't happen, but if it does then it is an error
            if(n_adults == 0 and n_seniors == 0):
                print("ERROR assignHomes() all adults and seniors already assigned!")
                print("        child_population= %d child_idx= %d" % (child_population, child_idx))
                break;

            
            # assign rest of member of household                
            for n in range(0, node_size-1):                            
                # adjust probability of finding child in home
                children_left = child_population - child_idx 
                adults_left = child_population - adult_idx

                child_adj_prob = 0.0

                # if two adults or seniors in house, then it is more likely
                # to have a child in house as oppposed to another adult or senior
                if(n_adults + n_seniors > 2):
                    child_adj_prob = 0.9
                elif(n_adults + n_seniors > 1):
                    child_adj_prob = 0.65
                    
                # if we are running out of adults to assign, then we need to start finding more children
                if(adults_left < 2*children_left):
                    child_adj_prob = 1.0
                elif(adults_left < 3*children_left):
                    child_adj_prob += 0.3
                    
                rnd = np.random.ranf()
                if(rnd < (child_pop_distr + child_adj_prob) and (child_idx < child_population)):
                    n_children += 1
                else:
                    # if there is adult in household, there is likely to be 
                    # another adult as opposed to another senior
                    if(n_adults > 0 and n_adults < 2):
                        rnd2 = np.random.ranf()
                        if(rnd2 < adult_senior_mix_prob):
                            n_seniors += 1
                        else:
                            n_adults += 1
                    elif(n_seniors > 0 and n_seniors < 2):
                        rnd2 = np.random.ranf()
                        if(rnd2 < adult_senior_mix_prob):
                            n_seniors += 1
                        else:
                            n_adults += 1
                # end for
            
            if(adult_idx + n_adults > adult_population):
                n_adults = adult_population - adult_idx
            if(child_idx + n_children > child_population):
                n_children = child_population - child_idx
            if(senior_idx + n_seniors > senior_population):
                n_seniors = senior_population - senior_idx 
                
            # assign 
            if(n_adults > 0):            
                adults = adult_perm[adult_idx:adult_idx+n_adults]
                self.home_node_prsn[adults] = home_idx
                self.node_schedule_prsn[adults, :, :] = home_idx  # default schedule is at home
            if(n_seniors > 0):                           
                seniors = senior_perm[senior_idx:senior_idx+n_seniors]
                self.home_node_prsn[seniors] = home_idx
                self.node_schedule_prsn[seniors, :, :] = home_idx  # default schedule is at home
            if(n_children > 0):                           
                children = child_perm[child_idx:child_idx+n_children]
                self.home_node_prsn[children] = home_idx
                self.node_schedule_prsn[children, :, :] = home_idx # default schedule is at home
                
            adult_idx += n_adults
            child_idx += n_children
            senior_idx += n_seniors
            
            self.home_node_num_children[home_idx] = n_children
            self.home_node_num_adults[home_idx] = n_adults
            self.home_node_num_seniors[home_idx] = n_seniors
            
            home_idx += 1
            
        # end while    
        ppl_assigned = child_idx + adult_idx + senior_idx
        if(ppl_assigned != population):
            print("ERROR people assigned does not match population! ppl_assigned= %d populatin= %d" % (ppl_assigned, population))
            print("   adult_population= %d adult_idx= %d" % (adult_population, adult_idx))
            print("   child_population= %d child_idx= %d" % (child_population, child_idx))
            print("   senior_population= %d senior_idx= %d" % (senior_population, senior_idx))
            sys.exit(-1)
            
    def assignWorkSchool(self):
        # work shift for the population,m expressed in times per day
        self.work_shift_begin = np.zeros(population)
        self.work_shift_end = np.zeros(population)
        
        working_population = adult_population 
        worker_perm = np.random.permutation(working_population)
        worker_perm += child_population
        worker_idx = 0
        
        # assign place of work for working population
        
        # first ensure schools and hospitals are staffed
        idx_begin = self.school_node_idx_begin
        idx_end = idx_begin + num_school_nodes + num_hospital_nodes
        for node_idx in range(idx_begin, idx_end):
            staff_sz = self.typical_node_size[node_idx]
            isHospital = node_idx in range(self.hospital_node_idx_begin, self.hospital_node_idx_begin + num_hospital_nodes)
            isSchool = node_idx in range(self.hospital_node_idx_begin, self.school_node_idx_begin + num_school_nodes)
            if isSchool:
                staff_sz = int(round(staff_sz * school_staff_ratio))
            elif isHospital:
                staff_sz = int(round(staff_sz * hospital_staff_ratio))
            else:  # shouldn't happen
                continue 
                    
            workers = worker_perm[worker_idx:worker_idx + staff_sz]
            self.work_school_node_prsn[workers] = node_idx
            
            # assign shift
            if isHospital:
                idx1 = staff_sz // 4
                idx2 = idx1 + staff_sz // 4
                idx3 = idx2 + staff_sz // 4 
                idx4 = idx3 + staff_sz // 4 
                
                workers1 = workers[0:idx1]
                workers2 = workers[idx1:idx2]
                workers3 = workers[idx2:idx3]
                workers4 = workers[idx3:idx4]
                workers5 = workers[idx4:staff_sz]
                
                self.node_schedule_prsn[workers1, 0:5, 1:2] = node_idx  # 9to5 type shift
                self.node_schedule_prsn[workers2, 0:5, 3:4] = node_idx  # evening shift
                self.node_schedule_prsn[workers3, 0:5, 0] = node_idx  # twighlit shift
                
                self.node_schedule_prsn[workers4, 0:5, 1] = node_idx  # weekend day
                self.node_schedule_prsn[workers5, 0:5, 2] = node_idx  # weekend day
                
                # weekends
                self.node_schedule_prsn[workers4, 5:7, 1:2] = node_idx  # weekend day
                self.node_schedule_prsn[workers3, 5:7, 3:4] = node_idx  # weekend evening
                self.node_schedule_prsn[workers5, 5:7, 0] = node_idx  # weekend twlight                                                    # weekend twlight                
                
            else: # school
                self.node_schedule_prsn[workers, 0:5, 1:2] = node_idx
            
            worker_idx += staff_sz
            node_idx += 1
                    
        # assign remaining workers to private work and retail in random order
        nWork = num_private_work_nodes + num_retail_nodes                    
        work_nodes = np.random.permutation(nWork)
        work_nodes += self.private_work_node_idx_begin
        
        n_work = 0
        n_work_nodes = len(work_nodes)
        while(worker_idx < working_population and n_work < n_work_nodes):
            node_idx = work_nodes[n_work] 
            staff_sz = self.typical_node_size[node_idx]
            isRetail = node_idx in range(self.retail_node_idx_begin, self.retail_node_idx_begin + num_retail_nodes)
            workers = worker_perm[worker_idx:worker_idx + staff_sz]

            if isRetail:
                idx1 = staff_sz // 3
                idx2 = idx1 + staff_sz // 3
                idx3 = idx2 + staff_sz // 3 
                workers1 = workers[0:idx1]
                workers2 = workers[idx1:idx2]
                workers3 = workers[idx2:idx3]
                workers4 = workers[idx3:staff_sz]
                self.node_schedule_prsn[workers1, 0:5, 1:2] = node_idx  # 9to5 type shift
                self.node_schedule_prsn[workers2, 0:5, 3:4] = node_idx  # evening shift
                self.node_schedule_prsn[workers3, 5:7, 1:2] = node_idx  # weeknds
                self.node_schedule_prsn[workers4, 5:7, 3:4] = node_idx
            else:  # private work
                self.node_schedule_prsn[workers, 0:5, 1:2] = node_idx                
                
            worker_idx += staff_sz        
            n_work += 1 
            
        if worker_idx < working_population-1:
            print("exceess workers: %d " % (working_population-worker_idx - 1,))
        if n_work < n_work_nodes-1:
            print("exceess workplaces: %d " % (n_work_nodes - n_work - 1,))
            
                    
        children_perm = np.random.permutation(child_population)
        child_idx = 0
        # assign schools for children
        for n_sch in range(0, num_school_nodes):
            node_idx = self.school_node_idx_begin + n_sch
            school_sz = self.typical_node_size[node_idx]
            #print("n_sch= %d child_idx= %d school_sz= %d" % (n_sch, child_idx, school_sz))

            children = children_perm[child_idx:child_idx+school_sz]
            self.work_school_node_prsn[children] = node_idx
            child_idx += school_sz       
            
            # assign schedule
            self.node_schedule_prsn[children, 0:5, 1:2] = node_idx
            
            
    def assignVisitNodes(self):
        self.visit_node_household = -1*np.ones([num_home_nodes, visit_node_max], dtype=np.int32)
        self.visit_node_prsn = -1*np.ones([population, visit_node_max], dtype=np.int32)
        
        # number of different retail that are common to household
        num_comm_rtl_essntl = 5   # essential retail (e.g, grocery) for household
        num_comm_rtl_other  = 5   # other retail (e.g, barber, restaraunts, etc)
        num_indiv_rtl = 5         # individaul retail nodes
        
        if(useHelperLibrary):
            ret = self.helperlib.assignVisitNodes(self.visit_node_household, self.visit_node_prsn,
                                                  visit_node_max, num_comm_rtl_essntl, 
                                                  num_comm_rtl_other, num_indiv_rtl)
            return 
            
        
        # base visit nodes on household
        # household members likley to visit similar places
        for hm_idx in range(0, num_home_nodes):
            essntl_rtl_idx = np.random.randint(0, self.num_essential_retail, num_comm_rtl_essntl)
            essntl_rtl = self.essential_retail_nodes[essntl_rtl_idx]
            
            non_esstl_rtl = np.random.randint(0, num_retail_nodes, num_comm_rtl_non_essntl)
            non_esstl_rtl += self.retail_node_idx_begin
            
            self.visit_node_household[hm_idx, 0:num_comm_rtl_essntl] = essntl_rtl
            idx1 = num_comm_rtl_essntl
            idx2 = idx1 + num_comm_rtl_non_essntl
            self.visit_node_household[hm_idx, idx1:idx2] = non_esstl_rtl
            
        for p in range(0, population):
            rtl_idx = np.random.randint(0, num_retail_nodes, num_indiv_rtl)
            rtl_idx = self.retail_node_idx_begin
            self.visit_node_prsn[p, 0:num_indiv_rtl] = rtl_idx
                
                
    def scheduleVisits(self):
        # copy base schedule
        self.node_schedule_prsn_temp[:, :, :] = self.node_schedule_prsn[:, :, :]
        
        if useHelperLibrary:
            ret = self.helperlib.scheduleVisits(self.visit_node_household, self.visit_node_prsn, 
                                                visit_node_max, visits_per_schedule)
            return ret 
        
        visit_nodes = np.zeros(visit_node_max*2, dtype=np.int32)
        for p in range(0, population):
            hm_idx = self.home_node_prsn[p]
            
            # select reasonable time frame
            schdl = self.node_schedule_prsn[p, :, 1:4]
                
            # when person scheduled is home (free time)
            off_tm = np.argwhere(schdl == hm_idx)
            
            num_off_tm = len(off_tm)
            
            off_perm = np.random.permutation(num_off_tm)
            
            hshld_visit = self.visit_node_household[hm_idx, :]
            # trucnate 
            a = np.argwhere(hshld_visit < 0) 
            if(a.shape[0] > 0):
                hshld_visit = hshld_visit[0:a.shape[0]]
            
            prsn_visit = self.visit_node_prsn[p, :]
            # truncate 
            a = np.argwhere(prsn_visit < 0) 
            if(a.shape[0] > 0):
                hshld_visit = prsn_visit[0:a.shape[0]]

            # combine houshold visit nodes with person visit nodes
            idx1 = len(hshld_visit)
            idx2 = idx1 + len(prsn_visit)
            visit_nodes[0:idx1] = hshld_visit
            visit_nodes[idx1:idx2] = prsn_visit
            
            nd_to_visit = np.random.randint(0, len(visit_nodes), len(visit_nodes))
            
            for n in range(0, visits_per_schedule):
                tm = off_tm[off_perm[n]]
                self.node_schedule_prsn[p, tm[0], tm[1]] = visit_nodes[nd_to_visit[n]]
            
        
    def initHomesAndWorkPlace(self):
        self.private_work_node_idx_begin = num_home_nodes
        self.retail_node_idx_begin = self.private_work_node_idx_begin + num_private_work_nodes
        self.school_node_idx_begin = self.retail_node_idx_begin + num_retail_nodes
        self.hospital_node_idx_begin = self.school_node_idx_begin + num_school_nodes

        # assign sizes of home by grouping by household size
        idx1 = 0
        for sz in range(0, household_size_max):
            idx2 = int(round(num_home_nodes*household_distr_cs[sz]))
            self.typical_node_size[idx1:idx2] = sz + 1
            self.max_node_size[idx1:idx2] = (sz+1) * 3
            self.node_type[idx1:idx2] = int(Node.home)
            idx1 = idx2
            
        # randomly permute the node sizes
        rnd_perm = np.random.permutation(num_home_nodes)
        self.typical_node_size[0:num_home_nodes] = self.typical_node_size[rnd_perm]
        self.max_node_size[0:num_home_nodes] = self.max_node_size[rnd_perm]
        
        # assign sizes of private work nodes
        n_nodes = num_private_work_nodes
        wk_size = private_work_sizes_avg
        wk_std = private_work_sizes_std
        node_sz = np.random.normal(wk_size, wk_std, n_nodes)
        node_sz = np.int32(node_sz)
        np.place(node_sz, node_sz < 2, 2);
        np.place(node_sz, node_sz > node_size_max, node_size_max);
        
        idx1 = self.private_work_node_idx_begin
        idx2 = idx1 + n_nodes
        self.typical_node_size[idx1:idx2] = node_sz
        # allow some visitors
        self.max_node_size[idx1:idx2] = 10 + self.typical_node_size[idx1:idx2]
        self.node_type[idx1:idx2] = int(Node.private_work)
        # make certain fraction essential 
        self.num_essential_private_work = int(round(n_nodes * esssential_private_fraction))
        self.essential_private_work_nodes =  np.random.randint(idx1, idx2+1, self.num_essential_private_work)
        
        # assign sizes of retail nodes
        n_nodes = num_retail_nodes
        rtl_size = retail_sizes_avg
        rtl_std = retail_sizes_std
        node_sz = np.random.normal(rtl_size, rtl_std, n_nodes)
        node_sz = np.int32(node_sz)
        np.place(node_sz, node_sz < 2, 2);
        np.place(node_sz, node_sz > node_size_max, node_size_max);
        
        idx1 = self.retail_node_idx_begin
        idx2 = idx1 + n_nodes
        self.typical_node_size[idx1:idx2] = node_sz
        # allow some visitors
        self.max_node_size[idx1:idx2] = 5 * self.typical_node_size[idx1:idx2]
        self.node_type[idx1:idx2] = int(Node.retail)
        
        # make certain fraction essential 
        self.num_essential_retail = int(round(n_nodes * esssential_retail_fraction))
        self.essential_retail_nodes = np.random.randint(idx1, idx2+1, self.num_essential_retail)
        
        # assign sizes of school nodes
        for n_sch in range(0, num_school_nodes):
            sch_size = school_size_avg
            idx1 = self.school_node_idx_begin + n_sch
            self.typical_node_size[idx1] = sch_size
            # allow some visitors
            self.max_node_size[idx1] = 5 + self.typical_node_size[idx1]
            self.node_type[idx1] = int(Node.school)
            
        # assign sizes of hospital nodes
        for n_hsp in range(0, num_hospital_nodes):
            hsp_size = hospital_size_avg
            idx1 = self.hospital_node_idx_begin + n_hsp
            self.typical_node_size[idx1] = hsp_size
            # allow many visitors for hospitals
            self.max_node_size[idx1] = 10*self.typical_node_size[idx1]
            self.node_type[idx1] = int(Node.hospital)


        np.place(self.typical_node_size, self.typical_node_size < 2, 2);
        np.place(self.typical_node_size, self.typical_node_size > node_size_max, node_size_max);
        np.place(self.max_node_size, self.max_node_size < 2, 2);
        np.place(self.max_node_size, self.max_node_size > node_size_max, node_size_max);
        
        print("node begin private_work= %d retail= %d school= %d" % (self.private_work_node_idx_begin, self.retail_node_idx_begin, self.school_node_idx_begin))
        
        bad_nodes = np.argwhere(self.typical_node_size <= 0)
        if(bad_nodes.shape[0] > 0):
            print("bad nodes! typical size <= 0 ", bad_nodes.shape[0])
            print("  nodes= %s" % (bad_nodes[0, :],))
        bad_nodes = np.argwhere(self.typical_node_size > 1000)
        if(bad_nodes.shape[0] > 0):
            print("bad nodes! typical size > 1000 ", bad_nodes.shape[0])
            print("  nodes= %s " % (bad_nodes[0, :],))
        bad_nodes = np.argwhere(self.max_node_size <= 0)
        if(bad_nodes.shape[0] > 0):
            print("bad nodes! max size <= 0 ",  bad_nodes.shape[0])
            print("  nodes= %s " % (bad_nodes[0, :],))
        bad_nodes = np.argwhere(self.max_node_size > 1000)
        if(bad_nodes.shape[0] > 0):
            print("bad nodes! max size > 1000 ", bad_nodes.shape[0])
            print("  nodes= %s " % (bad_nodes[0, :],))
            
    def __init__(self):
        t1 = time.time()
        
        
        # initialize ages
        # index: person ID
        # output age of person
        self.age_prsn = -1*np.ones(population, dtype=np.uint8)
        
        # home node of person
        # index: person ID
        self.home_node_prsn = -1*np.ones(population, dtype=np.int32)
       
        # primary work node (or school node for child)
        # index: person ID
        # output: non_home node or -1 for no work (retired)
        self.work_school_node_prsn = -1*np.ones(population, dtype=np.int32) 

        # schjdule for the person, determinging where a person should be at a given time
        # index1: person ID
        # index2: day in week
        # index3: time phase of dayh
        
        # the "typical" schedule, which does not include visits 
        self.node_schedule_prsn = -1*np.ones([population, schedule_days, time_phases_per_day], dtype=np.int32)
        
        # the "transient" schedule which is based on typical schedule, with visits to nodes besides home/work
        self.node_schedule_prsn_temp =  1*np.ones([population, schedule_days, time_phases_per_day], dtype=np.int32)                     
        
        # usual number of people occupying this node
        # for all home and non_home nodes
        self.typical_node_size = np.zeros(num_nodes, dtype=np.uint32)  
    
        # max number of people node allows, including visitors
        self.max_node_size = np.zeros(num_nodes, dtype=np.uint32)  
        
        # all nodes, work, home, school, retail, hospital, etc.
        # first index: node ID
        # second index: node slot
        # output ID of person, or -1 if unoccpies
        self.all_nodes = -1*np.ones([num_nodes, node_size_max], dtype=np.int32)


        # type of node 
        self.node_type = -1*np.ones(num_nodes, dtype=np.uint8)
        # of people currently occupying node
        self.current_node_size = np.zeros(num_nodes, dtype=np.uint32)  

        # where person is 
        # index: person ID
        # output 0 = home node 1 = nonhome_node
        self.current_prsn_loc = -1*np.ones(population, dtype=np.uint8)  

        # current node that person occupies
        # index: person ID 
        # output first index into home_nodes or non_home_nodes
        self.current_node_prsn = -1*np.ones(population, dtype=np.int32)
        
        # current node slot that person occupies
        # index: person ID  
        # output second index into all_nodes
        self.current_node_slot_prsn = -1*np.ones(population, dtype=np.int32)
        
        # infection state of the person
        # index: person ID  
        # output infection state
        self.infection_state = np.zeros(population, dtype=np.uint8)
        
        # start time of the infection in time units
        # index: person ID  
        # output infection start time of infection
        self.infection_start_time = -1*np.ones(population, dtype=np.int32)

        # start time of person becaomes to recovery in time units
        # index: person ID  
        # output infection start time of recovery 
        self.infection_recovery_start_time = -1*np.ones(population, dtype=np.int32)

        self.isNodeShutdown = np.zeros(population, dtype=np.bool)
        
        # assign ages by grouping age range
        # e.g. indexes 0 to 99 will beage 0, 100 to 199 age 1, etc.
        n_age_rngs = len(age_distr)
        idx1 = 0
        for n_age in range(0, n_age_rngs):
            idx2 = int(round(population*age_distr_cs[n_age]))
            self.age_prsn[idx1:idx2] = n_age
            idx1 = idx2 + 1
            
        print(self.age_prsn[0:20])
        
        self.initHomesAndWorkPlace();
       
        if useHelperLibrary:
            t1 = time.time()
            self.initHelperLibrary()
            tElapsed = time.time() - t1

        tElapsed = time.time() - t1
        print("initialize vars time %0.2f" % (tElapsed,))
        
        t1 = time.time()
        self.assignHomes()
        tElapsed = time.time() - t1
        print("assignHomes time %0.2f" % (tElapsed,))
        
        t1 = time.time()
        self.assignWorkSchool()
        tElapsed = time.time() - t1
        print("assignWorkSchool time %0.2f" % (tElapsed,))
        
        t1 = time.time()
        self.assignVisitNodes()
        tElapsed = time.time() - t1
        print("assignVisitNodes time %0.2f" % (tElapsed,))

        # people who are initially infected
        self.initial_infection_idx = np.random.randint(0, population, initial_infections)            
        
        t1 = time.time()
        self.reset()   
        tElapsed = time.time() - t1
        print("reset time %0.2f" % (tElapsed,))
        
        t1 = time.time()
        self.scheduleVisits()
        tElapsed = time.time() - t1
        print("scheduleVisits time %0.2f" % (tElapsed,))

         
    def spreadInfection(self, tm):
        if useHelperLibrary:
            ret = self.helperlib.spreadInfection(tm, infection_chance_per_time, infection_max_people_per_time)
            return ret
        
        inf_arry = infection_chance_per_time * np.ones(infection_max_people_per_time);

        for node_idx in range(0, num_nodes):
            sz = self.current_node_size[node_idx]
            # nothing to do - cannot infect if less than two people
            if(sz < 2): 
                continue
            
            # determine # of current infections
            node =  self.all_nodes[node_idx, :]
            ppl_idx = np.argwhere(node != -1)
            ppl_idx = ppl_idx[:, 0]
            ppl = node[ppl_idx]
            #print("node %d ppl= %s" % (node_idx, repr(ppl)))
            inf = self.infection_state[ppl]

            n_cntg = np.sum((inf >= DiseaseState.infected_contagious) * (inf <= DiseaseState.recovery_contagious))
            n_imm = np.sum(inf == DiseaseState.recovery_not_contagious)
            
            # no contagious infections, so nothing to do 
            if(n_cntg == 0):
                continue
            
            # entire location infected, recovered or dead
            if(n_cntg + n_imm == sz):
                continue

            # if(tm % 12 == 0):
            #     print('n= ' + repr(n) + ' sz=' + repr(sz) + ' nd=' + repr(nd) + ' inf=' + repr(inf) + ' n_cntg=' + repr(n_cntg))
#                print('n= %d n_inf= %d' % (n, n_inf))
            
            # determine new infections
            inf_num_ppl = infection_max_people_per_time
            inf_new = np.zeros(inf_num_ppl)
            
            # each infected individual has random chance to infect 
            # other individuals
            for k in range(0, n_cntg):
                # construct random list of people
                ppl_contact_idx = np.random.randint(0, sz, inf_num_ppl)
                
                rnd_f = np.random.ranf(inf_num_ppl)
                new_inf = rnd_f < inf_arry[0:inf_num_ppl]
                new_inf = np.ones(inf_num_ppl)*new_inf 
                
                ppl_contact = ppl[ppl_contact_idx]
                ppl_inf_state = self.infection_state[ppl_contact]
                # if(tm % 12 == 0):
                #     print('ppl_contact= ' + repr(ppl_contact))
                #     print('ppl_idx= ' + repr(ppl_idx))
                #     print('ppl_inf_state= ' + repr(ppl_inf_state))
                
                for j in range(0, inf_num_ppl):
                    if(new_inf[j] > 0 and ppl_inf_state[j] == DiseaseState.not_infected):
                        self.infection_state[ppl_contact[j]] = 1
                        self.infection_start_time[ppl_contact[j]] = tm

            
    def advanceInfectionState(self, tm, stats=None):
        if useHelperLibrary:
           ret = self.helperlib.advanceInfectionState(tm)
           return ret
           
        for p in range(0, population):            
            if(self.infection_state[p] in [DiseaseState.not_infected, DiseaseState.recovery_contagious, DiseaseState.recovery_not_contagious, DiseaseState.deceased]):
                continue 
            
            td = tm - self.infection_start_time[p] 
            inf_state = self.infection_state[p]
            new_inf_state = inf_state
            age = self.age_prsn[p]
 
            # infection state
            # 0 not infected
            # 1 infected incubation not contagious
            # 2 infected incubation contagious
            # 3 infected no/minor symptoms
            # 4 infected sick
            # 5 infected hospitalized
            # 6 recovered (not contagious)
            # 7 deceased        
            
            # print('n= %d state= %d td= %d' % (n, inf_state, td))
            if(inf_state == DiseaseState.infected_not_contagious and td >= contagious_time):
                self.infection_state[p] = DiseaseState.infected_contagious
                
            elif(inf_state == DiseaseState.infected_contagious and td >= incubation_time):
                rnd = np.random.ranf()
                if(rnd < minor_symptoms_fraction):
                    self.infection_state[p] = DiseaseState.minor_symptoms
                else:
                    self.infection_state[p] = DiseaseState.sick
                    
            elif(inf_state == DiseaseState.minor_symptoms and td > recovery_time_minor_symptoms):
                self.infection_state[p] = DiseaseState.recovery_contagious
                self.infection_recovery_start_time[p] = tm
                    
            elif(inf_state == DiseaseState.sick and td > hospitalization_time):
                rnd = np.random.ranf()
                if(rnd < hospitalization_fraction):
                    self.infection_state[p] = DiseaseState.hospitalized
                else:
                    self.infection_state[p] = DiseaseState.recovery_contagious
                    self.infection_recovery_start_time[p] = tm

            # hospitalization outcome                    
            elif(inf_state == DiseaseState.hospitalized):
                if(td == death_time):
                    # determine if individual will recover or die
                    mort = mortality_vs_age_hospitalization[age]
                    rnd = np.random.ranf()
                    if(rnd < mort):
                        self.infection_state[p] = DiseaseState.deceased
                elif(td >= recovery_time_hospitalization):
                    self.infection_state[p] = DiseaseState.recovery_contagious
                    self.infection_recovery_start_time[p] = tm
                    
            # recovery    
            elif(inf_state == DiseaseState.recovery_contagious):
                td_rcv = tm - self.infection_recovery_start_time[p]
                if(td_rcv >= recovery_time_contagious):
                    self.infection_state[p] = DiseaseState.recovery_not_contagious
                    

            if(stats != None):
                new_inf_state = self.infection_state[p]
                if(new_inf_state in range(DiseaseState.infected_not_contagious, DiseaseState.recovery_contagious+1)):
                    stats.inf[tm] += 1
                if(new_inf_state in range(DiseaseState.infected_contagious, DiseaseState.recovery_contagious+1)):
                    stats.cntgs[tm] += 1
                if(new_inf_state in range(DiseaseState.sick, DiseaseState.hospitalized+1)):
                    stats.sick[tm] += 1
                if(new_inf_state == DiseaseState.hospitalized):
                    stats.hosp[tm] += 1
                if(new_inf_state == DiseaseState.recovery_contagious or new_inf_state == DiseaseState.recovery_not_contagious):
                    stats.rec[tm] += 1
                if(new_inf_state == DiseaseState.deceased):
                    stats.dec[tm] += 1
      
    def movePopulation(self, t_day, t_phase):
        if(useHelperLibrary):
            #print(repr(self.all_nodes[200, 0:self.max_node_size[200]]))
            #print(repr(self.node_schedule_prsn[200, :, :]))
            ret = self.helperlib.movePopulation(t_day, t_phase, self.next_hospital_idx, self.isNodeShutdown)
            if(ret < 0):
                print('helper lib movePopulation() failed!')
            else:
                self.next_hospital_idx = ret 
                
            return
        
        for p in range(0, population):
            cur_node = self.current_node_prsn[p]
            cur_slt = self.current_node_slot_prsn[p]
            new_node = self.node_schedule_prsn_temp[p, t_day % schedule_days, t_phase]
            
            inf_state = self.infection_state[p]
            if(inf_state == DiseaseState.deceased):
                new_node = num_nodes + 1  

            # if person is sick, must move to home node
            if(inf_state == DiseaseState.sick):
                new_node = self.home_node_prsn[p]  
            # if person is hospitalized, must move to home node
            elif(inf_state == DiseaseState.hospitalized):
                # if already hospitalized and in hospital, then 
                if(self.current_prsn_loc[p] == 3):
                    continue
                
                # find hospital
                hosp_idx = self.next_hospital_idx
                n_hosp = 0
                while(self.current_node_size[hosp_idx] >= self.max_node_size[hosp_idx] and n_hosp < num_hospital_nodes):
                    hosp_idx += 1     
                    if(hosp_idx >= self.hospital_node_idx_begin + num_hospital_nodes):
                        hosp_idx = self.hospital_node_idx_begin
                    n_hosp += 1
                
                if(n_hosp >= num_hospital_nodes):
                    print("WARNING hospitals full!")
                    continue                
                new_node = hosp_idx
                
                hosp_idx += 1
                if(hosp_idx >= self.hospital_node_idx_begin + num_hospital_nodes):
                    hosp_idx = self.hospital_node_idx_begin
                self.next_hospital_idx = hosp_idx 
                
                self.current_prsn_loc[p] = 3
                
            if(cur_node < 0 or cur_slt < 0 or new_node < 0):
                print("ERROR bad node/slot p= %d cur_node= %d cur_slt= %d new_node= %d" % (p, cur_node, cur_slt, new_node))
                continue
            
            cur_size = self.current_node_size[cur_node]
            
            if(cur_size == 0):
                print("ERROR node size is 0! node= %d cur_size= %d" % (new_node, cur_size))
                break
                
                
            if(cur_node != new_node):
                # find first unoccopied slot
                if(new_node < num_nodes):
                    cur_size_new_node = self.current_node_size[new_node]
                    max_size_new_node = self.max_node_size[new_node]
                    if(cur_size_new_node >= max_size_new_node):
                        print("WARNING node size exceeded! node= %d cur_size= %d max_size= %d" % (new_node, cur_size_new_node, max_size_new_node))
                        continue
            
                    new_slt = np.argwhere(self.all_nodes[new_node, :] < 0)
                    if(new_slt.shape[0] < 1):
                        print("ERROR no empty slots p= %d new_node= %d" % (p, new_node))
                        continue
                    
                    new_slt = new_slt[0][0]  
                    self.all_nodes[new_node, new_slt] = p
                    self.current_node_size[new_node] += 1
                    self.current_node_prsn[p] = new_node
                    self.current_node_slot_prsn[p] = new_slt

                self.all_nodes[cur_node, cur_slt] = -1
                self.current_node_size[cur_node] -= 1
                
                if(new_node == self.home_node_prsn[p]):
                    self.current_prsn_loc[p] = 0
                elif(new_node == self.work_school_node_prsn[p]):
                    self.current_prsn_loc[p] = 1
                    
    # verifies that the variables are in conssistent state
    def verifyStateConsistency(self):   
        err = False
        
        # verify node sizes and people in node
        for n_nd in range(0, num_nodes):
            node = self.all_nodes[n_nd, :]
            a = np.argwhere(node >= 0)
            n_ppl = a.shape[0]
            sz = self.current_node_size[n_nd] 
            if(sz != n_ppl):
                print("ERROR node size inconsistency node= %d size= %d actual= %d" % (n_nd, sz, n_ppl))
                err = True
                break
                
            ppl = a[:, 0]
            for slt_idx in ppl:
                prsn = node[slt_idx]
                cur_node = self.current_node_prsn[prsn] 
                if(cur_node != n_nd):
                    print("ERROR person location inconsistency node= %d person= %d actual node= %d" % (n_nd, prsn, cur_node))
                    err = True
                    break
                    
                cur_slt = self.current_node_slot_prsn[prsn]
                if(cur_slt != slt_idx):
                    print("ERROR person location inconsistency node= %d person= %d slot= %d actual slot= %d" % (n_nd, prsn, cur_slt, slt_idx))
                    err = True
                    break
            if err:
                break
            #end for p
        #end for node
                    
        return err
    
    def adjustScheduleShutdown(self, isSchoolShutdown, isNonEssentialBusinessShutdown): 
        idx1 = self.school_node_idx_begin
        idx2 = idx1 + num_school_nodes
        self.isNodeShutdown[idx1:idx2] = isSchoolShutdown
            
        idx1 = self.private_work_node_idx_begin
        idx2 = idx1 + num_private_work_nodes
        self.isNodeShutdown[idx1:idx2] = isNonEssentialBusinessShutdown
        idx1 = self.retail_node_idx_begin
        idx2 = idx1 + num_retail_nodes
        self.isNodeShutdown[idx1:idx2] = isNonEssentialBusinessShutdown
        
        if(isNonEssentialBusinessShutdown):
            self.isNodeShutdown[self.essential_private_work_nodes] = False
            self.isNodeShutdown[self.essential_retail_nodes] = False
            
# simulation statistics 
class sim_stats:
    pass

def initStats(stats, tm_max):    
    nDays = tm_max / time_phases_per_day
    stats.tday = np.arange(0, nDays, 1 / time_phases_per_day)
    stats.inf = np.zeros(tm_max, dtype=np.uint32)
    stats.cntgs = np.zeros(tm_max, dtype=np.uint32)
    stats.sick = np.zeros(tm_max, dtype=np.uint32)
    stats.hosp = np.zeros(tm_max, dtype=np.uint32)
    stats.rec = np.zeros(tm_max, dtype=np.uint32)
    stats.dec = np.zeros(tm_max, dtype=np.uint32)
    
def calcStats(sim, stats, t_idx):
    inf_state = sim.infection_state
    stats.inf[t_idx] = np.sum((inf_state >= DiseaseState.infected_not_contagious) * (inf_state <= DiseaseState.hospitalized))
    stats.cntgs[t_idx] = np.sum((inf_state >= DiseaseState.infected_contagious) * (inf_state <= DiseaseState.recovery_contagious))
    stats.sick[t_idx] = np.sum((inf_state >= DiseaseState.sick) * (inf_state <= DiseaseState.hospitalized))
    stats.hosp[t_idx] = np.sum(inf_state == DiseaseState.hospitalized)
    stats.rec[t_idx] = np.sum(inf_state == DiseaseState.recovery_contagious) + np.sum(inf_state == DiseaseState.recovery_not_contagious)
    stats.dec[t_idx] = np.sum(inf_state == DiseaseState.deceased)
    
def printStats(sim, stats, t_phase, t_day):
    t_idx = t_day*time_phases_per_day + t_phase
    print('day= %d phase= %d infected= %d contagious= %d sick= %d hospialized= %d recovered= %d died= %d' % (t_day, t_phase, stats.inf[t_idx], stats.cntgs[t_idx], stats.sick[t_idx], stats.hosp[t_idx], stats.rec[t_idx], stats.dec[t_idx]))    
    
def saveStats(fName, stats):
    fp = open(fName, 'a')
    s = np.array2string(stats.tday, separator=',', max_line_width=10000 )  # convert to string
    fp.write('time(days):, ' + s[1:-1] + '\n')             # write removign commas at end

    s = np.array2string(stats.inf, separator=',', max_line_width=10000)  # convert to string
    fp.write('infected:, ' + s[1:-1] + '\n')             # write removign commas at end
    
    s = np.array2string(stats.cntgs, separator=',', max_line_width=10000)  # convert to string
    fp.write('contangious:, ' + s[1:-1] + '\n')            # write removign commas at end

    s = np.array2string(stats.sick, separator=',', max_line_width=10000)  # convert to string
    fp.write('sick:, ' + s[1:-1] + '\n')             # write removign commas at end

    s = np.array2string(stats.hosp, separator=',', max_line_width=10000)  # convert to string
    fp.write('hospitalized:, ' + s[1:-1] + '\n')             # write removign commas at end

    s = np.array2string(stats.rec, separator=',', max_line_width=10000)  # convert to string
    fp.write('recovered:, ' + s[1:-1] + '\n')             # write removign commas at end

    s = np.array2string(stats.dec, separator=',', max_line_width=10000)  # convert to string
    fp.write('deceased:, ' + s[1:-1] + '\n')             # write removign commas at end

    fp.close()
    

        
def run_sim(sim, adjustSched=False):
    t_day = 0
    t_phase = 0
    keepRunning = True
    isSchedAdj1 = True  # firs tschedule adjustment condition (e.g, school shutdown)
    isSchedAdj2 = True  # second schedule adjustment condition (e.g, non-essnetial business shutdown)
    stats = sim_stats()
    
    initStats(stats, timeSimDays*time_phases_per_day)
    while(keepRunning):
        tBegin = time.time()
        #err = sim.verifyStateConsistency()
        #if(err):
        #    break
        
        t_idx = t_day*time_phases_per_day + t_phase
        calcStats(sim, stats, t_idx)
        printStats(sim, stats, t_phase, t_day)

        tElapsed_stats = time.time() - tBegin
        #print(sim.current_prsn_loc[0:20])
        
        
        # at beginning of day move everyone to school or work
        t1 = time.time()
        sim.movePopulation(t_day, t_phase)
        tElapsed_movePopulation = time.time() - t1
        
        n_work = np.sum(sim.current_prsn_loc == 1)
        n_home = np.sum(sim.current_prsn_loc == 0)
        n_other = population - n_work - n_home
        print('people at work= %d home= %d other= %d' %  (n_work, n_home, n_other))
       
        t1 = time.time()
        sim.spreadInfection(t_idx)
        tElapsed_spreadInf = time.time() - t1
        #print("spreadInfection took %f seconds" % (tElapsed,))

        t1 = time.time()
        sim.advanceInfectionState(t_idx)
        tElapsed_advanceInf = time.time() - t1
#        print("advanceInfectionState took %f seconds" % (tElapsed,))            
        tElapsed = time.time() - tBegin
        print("times stats: %0.2f movePopulation: %0.2f spreadInfection: %0.2f advanceInfection: %0.2f tElapsed: %0.2f" % (tElapsed_stats, tElapsed_movePopulation, tElapsed_spreadInf, tElapsed_advanceInf, tElapsed))            
            
        # check fo rshut down conditions if not alreayd reached
        if adjustSched:
            if isSchedAdj1 and stats.sick[t_idx] > schoolShutdownCount:
                    print("shutting down schools... ", end='')
                    t1 = time.time()
                    sim.adjustScheduleShutdown(True, False)
                    tElapsed = time.time() - tBegin
                    print("timeElapsed = %0.2f" % (tElapsed,))
                    isSchedAdj1 = False
            elif isSchedAdj2 and stats.sick[t_idx] > nonEssentialBusinessShutdownCount:
                    print("shutting down non-essential business... ", end='')
                    t1 = time.time()
                    sim.adjustScheduleShutdown(True, True)
                    tElapsed = time.time() - tBegin
                    print("timeElapsed = %0.2f" % (tElapsed,))
                    isSchedAdj2 = False
        
        t_phase += 1
        if(t_phase == time_phases_per_day):
            t_phase = 0
            t_day += 1        
            
        if(t_day == timeSimDays):
            keepRunning = False    

                
    return stats

    
if __name__ == '__main__':
    sim = epi_sim()
    dt_now = datetime.datetime.now()
    
    
    date_time = dt_now.strftime("%Y_%m_%d_%H_%M_%S")
    for n in range(0, numRuns):
        stats1 = run_sim(sim)
        saveStats('stats1_' + date_time + '.csv', stats1)
        sim.reset()
    
    # run schdule a second time with shutdown condition
    nonEssentialBusinessShutdownCount = population
    schoolShutdownCount = 50
    for n in range(0, numRuns):
        stats2 = run_sim(sim, True)
        saveStats('stats2_' + date_time + '.csv', stats2)
        sim.reset()
        
    # run schdule a third time with more stringent shutdown condition
    nonEssentialBusinessShutdownCount = 200
    schoolShutdownCount = 50
    for n in range(0, numRuns):
        stats3 = run_sim(sim, True)
        saveStats('stats3_' + date_time + '.csv', stats3)
        sim.reset()
        
    nonEssentialBusinessShutdownCount = 50
    schoolShutdownCount = 10
    for n in range(0, numRuns):
        stats4 = run_sim(sim, True)
        saveStats('stats4_' + date_time + '.csv', stats4)
        sim.reset()

    # plotData(stats1, stats2)
    






