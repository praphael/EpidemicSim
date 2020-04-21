// dllmain.cpp : Defines the entry point for the DLL application.
#include "pch.h"

#include <stdio.h>
#include <vector>
#include <time.h>
#include <math.h>

#define _DLLEXPORT __declspec(dllexport)

#define MUTEX_WAIT_TIMEOUT_MS 500 

extern "C" {

    BOOL APIENTRY DllMain(HMODULE hModule,
        DWORD  ul_reason_for_call,
        LPVOID lpReserved
    )
    {
        switch (ul_reason_for_call)
        {
        case DLL_PROCESS_ATTACH:
        case DLL_THREAD_ATTACH:
        case DLL_THREAD_DETACH:
        case DLL_PROCESS_DETACH:
            break;
        }
        return TRUE;
    }

    int g_population;
    int g_numNodes;
    int g_maxSlotsPerNode;
    int g_scheduleDays;
    int g_timePhasesPerDay;

    int g_numHomeNodes;
    int g_numHospitalNodes;
    int g_numRetailNodes;
    int g_numRetailEssential;int g_hospitalNodeIndexBegin;
    int g_retailNodeIndexBegin;

    int* g_all_nodes;
    int* g_home_node_prsn;
    int* g_work_school_node_prsn;
    int* g_current_node_prsn;
    int* g_current_node_slot_prsn;
    int* g_node_schedule_prsn;
    char* g_infection_state;
    char* g_current_prsn_loc;
    int* g_current_node_size;
    UINT* g_max_node_size;
    int* g_infection_start_time;;
    char* g_age_prsn;
    int* g_infection_recovery_start_time;
    char* g_node_type;

    int g_tDay;
    int g_tPhase;
    int g_next_hospital_idx;
    bool g_isInitialized = false;
    int g_numThreads = 4;
    std::vector<HANDLE> g_nodeMutex;
    HANDLE g_hospitalMutex;

    int g_timeIdx;
    float g_infection_chance_per_time;
    int g_infection_max_people_per_time;

    // whether node is shutdown
    bool* g_isNodeShutdown;

    // diseas parameters
    int g_contagious_time;
    int g_incubation_time;
    float g_minor_symptoms_fraction;
    int g_recovery_time_minor_symptoms;
    int g_recovery_time_contagious;
    int g_recovery_time_hospitalization;
    float g_hospitalization_fraction;
    int g_hospitalization_time;
    int g_death_time;
    float* g_mortality_vs_age_hospitalization;

    const int NotInfected = 0;
    const int InfectedNotContagious = 1;
    const int InfectedContagious = 2;
    const int MinorSymptoms = 3;
    const int Sick = 4;
    const int Hospitalized = 5;
    const int RecoveryContagious = 6;
    const int RecoveryNotContagious = 7;
    const int Deceased = 8;

    const int MaxThreads = 32;

    const int TimeBusinessDayStart = 1;  // start time phase of business day (i.e, morning)
    const int TimeBusinessDayEnd = 4;    // end time phase of business day (i.e, late evening)

    DWORD movePopulationThread(LPVOID lParam) {
        int pBegin = 0;
        int pEnd = g_population;
        int del_p = g_numThreads;

        pBegin = *(int*)lParam;
        // printf("**helper_lib** pbegin= %d\n", pBegin);

        for (int p = pBegin; p < pEnd; p += del_p) {
            int cur_node = g_current_node_prsn[p];
            int cur_slt = g_current_node_slot_prsn[p];
            int sch_idx = p * g_scheduleDays * g_timePhasesPerDay + (g_tDay % g_scheduleDays) * g_timePhasesPerDay + g_tPhase;
            int new_node = g_node_schedule_prsn[sch_idx];
            int new_slt = -1;

            if (g_isNodeShutdown[new_node])
                new_node = g_home_node_prsn[p];

            char inf_state = g_infection_state[p];
            if (inf_state == Deceased) {
                new_node = g_numNodes + 1;
            }

            // if person is sick, must move to home node
            if (inf_state == Sick) {
                new_node = g_home_node_prsn[p];
            }
            // if person is hospitalized, must move to hospital node
            else if (inf_state == Hospitalized) {
                // if already hospitalized and in hospital, then do nothing
                if (g_current_prsn_loc[p] == 3) {
                    // printf("p=%d hospitalized\n", p);
                    continue;
                }

                DWORD err = ::WaitForSingleObject(g_hospitalMutex, MUTEX_WAIT_TIMEOUT_MS);
                if (err) {
                    printf("ERROR **helper_lib** hospital mutex wait timeout! p=%d node=%d\n", p, new_node);
                    continue;
                }

                // otherwise find hospital
                int hosp_idx = g_next_hospital_idx;
                int n_hosp = 0;
                while ((g_current_node_size[hosp_idx] >= g_max_node_size[hosp_idx]) && (n_hosp < g_numHospitalNodes)) {
                    hosp_idx += 1;
                    n_hosp += 1;
                    if (hosp_idx >= g_hospitalNodeIndexBegin + g_numHospitalNodes) {
                        hosp_idx = g_hospitalNodeIndexBegin;
                    }
                }

                if (n_hosp >= g_numHospitalNodes) {
                    printf("WARNING **helper_lib** hospitals full!\n");
                    ::ReleaseMutex(g_hospitalMutex);
                    continue;
                }
                new_node = hosp_idx;

                hosp_idx += 1;
                if (hosp_idx >= g_hospitalNodeIndexBegin + g_numHospitalNodes) {
                    hosp_idx = g_hospitalNodeIndexBegin;
                }
                g_current_prsn_loc[p] = 3;

                ::ReleaseMutex(g_hospitalMutex);
            }

            if (cur_node < 0 or cur_slt < 0 or new_node < 0) {
                printf("ERROR **helper_lib** bad node/slot p= %d cur_node= %d cur_slt= %d new_node= %d\n", p, cur_node, cur_slt, new_node);
                continue;
            }
            // don't have to do anything
            if (cur_node == new_node) {
                continue;
            }

            HANDLE rowMutex = g_nodeMutex[cur_node];
            HANDLE newRowMutex = NULL;
            if ((new_node > 0) && (new_node < g_numNodes))
                newRowMutex = g_nodeMutex[new_node];
            // obtain locks on both current and new node
            // to prevent deadlock, if we cannoot obtain both locks, 
            // release locks sleep for a bit and then try again
            DWORD err = 1;
            double tElapsed_ms = 0;
            clock_t clk = clock();
            while (err && tElapsed_ms < MUTEX_WAIT_TIMEOUT_MS) {
                // obtain a lock on current node
                err = ::WaitForSingleObject(rowMutex, 1);
                if (err) {
                    Sleep(5);
                    continue;
                }

                if (newRowMutex != NULL) {
                    // obtain a lock on new node
                    // if we cannot obtain lock due to timeout, then release lock 
                    // and try again next loop iteration
                    err = ::WaitForSingleObject(newRowMutex, 1);
                    if (err) {
                        ::ReleaseMutex(rowMutex);
                        Sleep(5);
                    }
                }

                tElapsed_ms = 1e3 * (clock() - clk) / (double)CLOCKS_PER_SEC;
            }
            if (err) {
                printf("ERROR **helper_lib** could not obtain locks on node p= %d cur_node= %d new_node= %d\n", p, cur_node, new_node);
                continue;
            }

            int cur_size = g_current_node_size[cur_node];

            if (cur_size == 0) {
                printf("ERROR **helper_lib** node size is 0! node= %d cur_size= %d\n", new_node, cur_size);
                ::ReleaseMutex(rowMutex);
                if (newRowMutex != NULL)
                    ::ReleaseMutex(newRowMutex);
                continue;
            }

            // find first unoccopied slot
            if (new_node < g_numNodes) {
                new_slt = 0;
                int new_node_size = g_current_node_size[new_node];
                int new_node_max_size = g_max_node_size[new_node];

                if (new_node_size >= new_node_max_size) {
                    // printf("WARNING **helper_lib** node size exceeded! node= %d size= %d max= %d\n", new_node, new_node_size, new_node_max_size);
                    ::ReleaseMutex(rowMutex);
                    if (newRowMutex != NULL)
                        ::ReleaseMutex(newRowMutex);
                    continue;
                }

                // new_slt = np.argwhere(self.all_nodes[new_node, :] < 0)
                for (new_slt = 0; new_slt < new_node_max_size; new_slt++) {
                    int idx = new_node * g_maxSlotsPerNode + new_slt;
                    int prsn = g_all_nodes[idx];
                    if (prsn == -1) break;
                }

                //  printf("p=%d cur_node= %d cur_slt= %d new_node=%d\n", p, cur_node, cur_slt, new_node);
                if (new_slt >= new_node_max_size) {
                    printf("ERROR **helper_lib** no empty slots p= %d new_node= %d cur_size= %d max_size= %d\n", p, new_node, new_node_size, new_node_max_size);
                    printf("\t");
                    for (new_slt = 0; new_slt < new_node_max_size; new_slt++) {
                        int idx = new_node * g_maxSlotsPerNode + new_slt;
                        int prsn = g_all_nodes[idx];
                        printf("%d ", prsn);
                    }
                    printf("\n");
                    ::ReleaseMutex(rowMutex);
                    ::ReleaseMutex(newRowMutex);
                    continue;
                }

                int idx = new_node * g_maxSlotsPerNode + new_slt;
                g_all_nodes[idx] = p;
                g_current_node_size[new_node] += 1;
                g_current_node_slot_prsn[p] = new_slt;
            }

            g_current_node_prsn[p] = new_node;
            int idx = cur_node * g_maxSlotsPerNode + cur_slt;
            if (g_all_nodes[idx] != -1) {
                g_all_nodes[idx] = -1;
                g_current_node_size[cur_node] -= 1;
            }

            if (new_node == g_home_node_prsn[p]) {
                g_current_prsn_loc[p] = 0;
            }
            else {
                if (new_node == g_work_school_node_prsn[p])
                    g_current_prsn_loc[p] = 1;
                else
                    g_current_prsn_loc[p] = 2;
            }

            if (inf_state == Hospitalized) {
                g_current_prsn_loc[p] = 3;
            }
            else if (inf_state == Deceased) {
                g_current_prsn_loc[p] = 4;
            }
            ::ReleaseMutex(rowMutex);
            if (newRowMutex != NULL)
                ::ReleaseMutex(newRowMutex);
        } // end for 

        return 0;
    }

    _DLLEXPORT int SetParams(int population,
        int numNodes,
        int maxSlotsPerNode,
        int scheduleDays,
        int timePhasesPerDay)
    {
        g_population = population;
        g_numNodes = numNodes;
        g_maxSlotsPerNode = maxSlotsPerNode;
        g_scheduleDays = scheduleDays;
        g_timePhasesPerDay = timePhasesPerDay;

        return 0;
    }

    _DLLEXPORT int SetNodeParams(int numHomeNodes,
                                 int numHospitalNodes,
                                 int numRetailNodes,
                                 int numRetailEssential,
                                 int hospitalNodeIndexBegin,
                                 int retailNodeIndexBegin) 
    {
        g_numHomeNodes = numHomeNodes;
        g_numHospitalNodes = numHospitalNodes;
        g_numRetailNodes = numRetailNodes;
        g_numRetailEssential = numRetailEssential;
        g_hospitalNodeIndexBegin = hospitalNodeIndexBegin;
        g_retailNodeIndexBegin = retailNodeIndexBegin;

        return 0;
    }

    _DLLEXPORT int SetDiseaseParams(int contagious_time,
        int incubation_time,
        float minor_symptoms_fraction,
        int recovery_time_minor_symptoms,
        int recovery_time_contagious,
        int recovery_time_hospitalization,
        int hospitalization_time,
        float hospitalization_fraction,
        int death_time,
        float* mortality_vs_age_hospitalization) {
        g_contagious_time = contagious_time;
        g_incubation_time = incubation_time;
        g_minor_symptoms_fraction = minor_symptoms_fraction;
        g_recovery_time_minor_symptoms = recovery_time_minor_symptoms;
        g_recovery_time_contagious = recovery_time_contagious;
        g_recovery_time_hospitalization = recovery_time_hospitalization;
        g_hospitalization_time = hospitalization_time;
        g_death_time = death_time;
        g_mortality_vs_age_hospitalization = mortality_vs_age_hospitalization;
        g_hospitalization_fraction = hospitalization_fraction;

        return 0;
    }

    _DLLEXPORT int SetArrays(int* all_nodes,
        int* home_node_prsn,
        int* work_school_node_prsn,
        int* current_node_prsn,
        int* current_node_slot_prsn,
        int* node_schedule_prsn,
        char* infection_state,
        char* current_prsn_loc,
        int* current_node_size,
        UINT* max_node_size,
        char* age_prsn,
        int* infection_start_time,
        int* infection_recovery_start_time)
    {
        g_all_nodes = all_nodes;
        g_home_node_prsn = home_node_prsn;
        g_work_school_node_prsn = work_school_node_prsn;
        g_current_node_prsn = current_node_prsn;
        g_node_schedule_prsn = node_schedule_prsn;
        g_current_node_slot_prsn = current_node_slot_prsn;
        g_infection_state = infection_state;
        g_current_prsn_loc = current_prsn_loc;
        g_current_node_size = current_node_size;
        g_max_node_size = max_node_size;
        g_age_prsn = age_prsn;
        g_infection_start_time = infection_start_time;
        g_infection_recovery_start_time = infection_recovery_start_time;

        return 0;
    }


    _DLLEXPORT int SetOptions(int numThreads) {
        if (numThreads > MaxThreads)
            numThreads = MaxThreads;
        g_numThreads = numThreads;

        return 0;
    }

    _DLLEXPORT int movePopulation(int t_day, int t_phase, int nextHospitalIdx, bool *isNodeShutdown)
    {
        g_tDay = t_day;
        g_tPhase = t_phase;
        g_next_hospital_idx = nextHospitalIdx;
        g_isNodeShutdown = isNodeShutdown;

        // printf("movePopulation: t_day= %d t_phase= %d nextHospitalIdx= %d\n", t_day, t_phase, nextHospitalIdx);

        // initalize mutexes
        if (!g_isInitialized) {
            g_nodeMutex.clear();
            g_nodeMutex.reserve(g_numNodes);
            for (int n = 0; n < g_numNodes; n++) {
                HANDLE mutexHnd = ::CreateMutex(NULL, FALSE, NULL);
                if (mutexHnd == NULL) {
                    printf("ERROR **helper lib** Failure to create node mutex %d!\n", n);
                    break;
                }
                else {
                    ::ReleaseMutex(mutexHnd);
                    g_nodeMutex.push_back(mutexHnd);
                }
            }

            g_hospitalMutex = ::CreateMutex(NULL, FALSE, NULL);
            if (g_hospitalMutex == NULL) {
                printf("ERROR **helper lib**  Failure to create hospital mutex!\n");
            }
            else {
                ::ReleaseMutex(g_hospitalMutex);
                g_isInitialized = true;
            }
        }
        if (!g_isInitialized) {
            printf("ERROR **helper lib** Initializtion failure!\n");
            return -1;
        }

        /*
        int nd = 200;
        int nd_size = g_current_node_size[nd];
        int mx_size = g_max_node_size[nd];
        printf("node %d size= %d max= %d\n\t", nd, nd_size, mx_size);
        for (int n = 0; n < mx_size; n++) {
            int idx = g_maxSlotsPerNode * nd + n;
            int prsn = g_all_nodes[idx];
            printf("%d ", prsn);
        }

        int p = 200;
        printf("person %d schedule \n\t", nd, nd_size, mx_size);
        for (int i = 0; i < g_scheduleDays; i++) {
            for (int j = 0; j < g_timePhasesPerDay; j++) {
                int idx = p * g_scheduleDays * g_timePhasesPerDay + i * g_timePhasesPerDay + j;
                int nd = g_node_schedule_prsn[idx];
                printf("%d ", nd);
            }
            printf("\n\t");
        }

        return 0;

        printf("max node size= ");
        for (int n = 0; n < g_numNodes; n++) {
            printf("%d ", g_max_node_size[n]);
        }
        printf("\n");
        */


        HANDLE threadHandles[32];
        int threadIdx[32];

        for (int nThd = 0; nThd < g_numThreads; nThd++) {
            DWORD thdID;
            threadIdx[nThd] = nThd;
            HANDLE thdHnd = ::CreateThread(NULL, 0, movePopulationThread, &threadIdx[nThd], 0, &thdID);
            if (thdHnd == NULL) {
                printf("ERROR **helper lib**  Failure to create thread %d!\n", nThd);
            }
            threadHandles[nThd] = thdHnd;
        }

        // wait for threads to finish
        ::WaitForMultipleObjects(g_numThreads, threadHandles, TRUE, 10000);

        return g_next_hospital_idx;
    }

    _DLLEXPORT int advanceInfectionState(int tm) {
        for (int p = 0; p < g_population; p++) {
            char inf_state = g_infection_state[p];

            if (inf_state > Deceased) {
                printf("ERROR Bad infection state! p= %d inf_state= %d", p, inf_state);
                continue;
            }

            if ((inf_state == NotInfected) || (inf_state == RecoveryContagious)
                || (inf_state == RecoveryNotContagious) || (inf_state == Deceased))
                continue;

            int td = tm - g_infection_start_time[p];
            auto new_inf_state = inf_state;
            auto age = g_age_prsn[p];

            //  printf('n= %d state= %d td= %d' % (n, inf_state, td))
            if ((inf_state == InfectedNotContagious) && (td >= g_contagious_time)) {
                g_infection_state[p] = InfectedContagious;
            }
            else if ((inf_state == InfectedContagious) && (td >= g_incubation_time)) {
                float rnd = (rand() % 10000) / (float)10000;
                if (rnd < g_minor_symptoms_fraction)
                    g_infection_state[p] = MinorSymptoms;
                else
                    g_infection_state[p] = Sick;
            }

            else if ((inf_state == MinorSymptoms) && (td > g_recovery_time_minor_symptoms)) {
                g_infection_state[p] = RecoveryContagious;
                g_infection_recovery_start_time[p] = tm;
            }
            else if ((inf_state == Sick) && (td >= g_hospitalization_time)) {
                float rnd = (rand() % 10000) / (float)10000;
                if (rnd < g_hospitalization_fraction)
                    g_infection_state[p] = Hospitalized;
                else {
                    g_infection_state[p] = RecoveryContagious;
                    g_infection_recovery_start_time[p] = tm;
                }
            }

            // hospitalization outcome
            else if (inf_state == Hospitalized) {
                if (td == g_death_time) {
                    // determine if individual will recover or die
                    float mort = g_mortality_vs_age_hospitalization[age];
                    float rnd = (rand() % 10000) / (float)10000;
                    if (rnd < mort) {
                        g_infection_state[p] = Deceased;
                    }
                }
                else if (td >= g_recovery_time_hospitalization) {
                    g_infection_state[p] = RecoveryContagious;
                    g_infection_recovery_start_time[p] = tm;
                }
            }
            //  recovery
            else if (inf_state == RecoveryContagious) {
                int td_rcv = tm - g_infection_recovery_start_time[p];
                if (td_rcv >= g_recovery_time_contagious)
                    g_infection_state[p] = RecoveryNotContagious;
            }
        }

        return 0;
    }

    _DLLEXPORT int arrayTest(int v[], int len) {
        for (int n = 0; n < len; n++) {
            v[n] *= 2;
            printf("%d ", v[n]);
        }
        printf("\n");

        return 0;
    }

    _DLLEXPORT int arrayTest2D(int v[], int d1, int d2) {
        for (int i = 0; i < d1; i++) {
            for (int j = 0; j < d2; j++) {
                int idx = i * d2 + j;
                v[idx] = i * 10 + j;
                printf("%d ", v[idx]);
            }
            printf("\n");
        }
        printf("\n");

        return 0;
    }

    _DLLEXPORT int arrayTest3D(int v[], int d1, int d2, int d3) {
        for (int i = 0; i < d1; i++) {
            for (int j = 0; j < d2; j++) {
                for (int k = 0; k < d3; k++) {
                    int idx = i * d2 * d3 + j * d3 + k;
                    v[idx] = k + j * 10 + i * 100;
                    printf("%d ", v[idx]);
                }
                printf("\n");
            }
            printf("\n");
        }
        printf("\n");

        return 0;
    }


    _DLLEXPORT DWORD spreadInfectionThread(LPVOID param) {
        std::vector<int> ppl;

        auto d_node = g_numThreads;
        auto node_begin = *(int*)param;

        for (int node_idx = node_begin; node_idx < g_numNodes; node_idx += d_node) {
            int sz = g_current_node_size[node_idx];
            // nothing to do - cannot infect if less than two people
            if (sz < 2)
                continue;

            bool isHospital = false;
            if (node_idx >= g_hospitalNodeIndexBegin && node_idx < g_hospitalNodeIndexBegin + g_numHospitalNodes)
                isHospital = true;

            ppl.clear();
            int n_cntg = 0;
            int n_not_inf = 0;
            char inf_state;
            // determine # of current infections
            for (int slt = 0; slt < g_maxSlotsPerNode; slt++) {
                auto idx = g_maxSlotsPerNode * node_idx + slt;
                auto prsn = g_all_nodes[idx];
                if (prsn != -1) {
                    ppl.push_back(prsn);
                    inf_state = g_infection_state[prsn];
                    if (inf_state == NotInfected) {
                        n_not_inf += 1;
                    }
                    else {
                        if ((inf_state == InfectedContagious) || (inf_state == MinorSymptoms)
                            || (inf_state == Sick) || (inf_state == RecoveryContagious)) {
                            n_cntg += 1;
                        }
                    }
                }
                // print("node %d ppl= %s" % (node_idx, repr(ppl)))
            }


            // no contagious infections, so nothing to do
            if (n_cntg == 0)
                continue;

            // everyone infected, nothing to do 
            if (n_not_inf == 0)
                continue;

            // determine new infections
            auto inf_num_ppl = g_infection_max_people_per_time;

            //  each infected individual has random chance to infect
            //  other individuals
            for (int k = 0; k < n_cntg; k++) {

                // 
                for (int n = 0; n < inf_num_ppl; n++) {
                    int rnd_idx = rand() % sz;
                    if (rnd_idx < ppl.size()) {
                        auto prsn = ppl[rnd_idx];
                        float rnd = (rand() % 10000) / (float)10000;
                        if (rnd < g_infection_chance_per_time) {
                            if (g_infection_state[prsn] == NotInfected) {
                                g_infection_state[prsn] = InfectedNotContagious;
                                g_infection_start_time[prsn] = g_timeIdx;
                            }
                        }
                    }
                    else {
                        printf("spreadInfection: index out of range! node_idx= %d sz= %d rnd_idx= %d", node_idx, sz, rnd_idx);
                    }

                }
            }
        } // end for

        return 0;
    }

    _DLLEXPORT int spreadInfection(int tm, float infection_chance_per_time, int infection_max_people_per_time) {
        g_timeIdx = tm;
        g_infection_chance_per_time = infection_chance_per_time;
        g_infection_max_people_per_time = infection_max_people_per_time;

        HANDLE threadHandles[32];
        int threadIdx[32];

        for (int nThd = 0; nThd < g_numThreads; nThd++) {
            DWORD thdID;
            threadIdx[nThd] = nThd;
            HANDLE thdHnd = ::CreateThread(NULL, 0, spreadInfectionThread, &threadIdx[nThd], 0, &thdID);
            if (thdHnd == NULL) {
                printf("ERROR **helper lib**  Failure to create thread %d!\n", nThd);
            }
            threadHandles[nThd] = thdHnd;
        }

        // wait for threads to finish
        ::WaitForMultipleObjects(g_numThreads, threadHandles, TRUE, 10000);

        return 0;
    }

    _DLLEXPORT int scheduleVisits(int* visit_node_household, 
                                  int* visit_node_prsn, 
                                  int visit_node_max, 
                                  int visits_per_schedule) 
    {
        // visit_nodes = np.zeros(visit_node_max * 2, dtype = np.int32);
        std::vector<int> visit_nodes;
        std::vector<int> off_tm;
        for (int p = 0; p < g_population; p++) {
            int hm_idx = g_home_node_prsn[p];

            // select reasonable time frame
            off_tm.clear();
            for (int t_dy = 0; t_dy < g_scheduleDays; t_dy++) {
                for (int t_ph = TimeBusinessDayStart; t_ph < TimeBusinessDayEnd; t_ph++) {
                    int idx = p * g_scheduleDays * g_timePhasesPerDay + t_dy * g_timePhasesPerDay + t_ph;
                    if (g_node_schedule_prsn[idx] != hm_idx)
                        off_tm.push_back(t_dy * g_timePhasesPerDay + t_ph);
                }
            }


            int num_off_tm = off_tm.size();
            // randomly permute times
            for (int n = 0; n < num_off_tm; n++) {
                int rnd_idx = rand() % num_off_tm;
                int tmp = off_tm[n];
                off_tm[n] = off_tm[rnd_idx];
                off_tm[rnd_idx] = tmp;
            }

            // do not schedule > 50% of off time
            if (visits_per_schedule > (num_off_tm / 2))
                visits_per_schedule = num_off_tm / 2;

            // add all visit nodes 
            int idx = 0;
            visit_nodes.clear();
            for (int n = 0; n < visit_node_max; n++) {
                int idx = visit_node_max * hm_idx + n;

                int nd = visit_node_household[idx];
                if (nd >= 0)
                    visit_nodes.push_back(nd);

                int nd2 = visit_node_prsn[idx];
                if (nd2 >= 0)
                    visit_nodes.push_back(nd2);

                // no more visit nodes so we can break out of the loop
                if (nd < 0 && nd2 < 0)
                    break;
            }
            int n_visit_nodes = visit_nodes.size();

            for (int v=0; v < visits_per_schedule; v++) {
                auto tm = off_tm[v];

                int t_dy = tm / g_timePhasesPerDay;
                int t_ph = tm % g_scheduleDays;
                int schd_idx = p * g_scheduleDays * g_timePhasesPerDay + t_dy * g_timePhasesPerDay + t_ph;
                int rnd_idx = rand() % n_visit_nodes;
                g_node_schedule_prsn[schd_idx] = visit_nodes[rnd_idx];
            }
        }

        return 0;
    }


    _DLLEXPORT int assignVisitNodes(int* visit_node_household, 
                                    int* visit_node_prsn,
                                    int visit_node_max, 
                                    int num_comm_rtl_esntl,
                                    int num_comm_rtl_other,
                                    int num_indiv_rtl)
    {
        std::vector<int> essntl_rtl;
        std::vector<int> other_rtl;

        // base visit nodes on household
        // household members likley to visit similar places
        for (int hm_idx = 0; hm_idx < g_numHomeNodes; hm_idx++) {
            essntl_rtl.clear();
            other_rtl.clear();
            int idx = hm_idx * visit_node_max;

            // generate random essential retail nodes
            for (int n = 0; n < num_comm_rtl_esntl; n++) {
                int rnd_idx = rand() % g_numRetailEssential;
                rnd_idx += g_retailNodeIndexBegin;
                visit_node_household[idx] = rnd_idx;
                idx++;
            }

            // generate random other retail
            for (int n = 0; n < num_comm_rtl_esntl; n++) {
                int rnd_idx = rand() % g_numRetailNodes;
                rnd_idx += g_retailNodeIndexBegin;
                visit_node_household[idx] = rnd_idx;
            }
        }

        for (int p = 0; p < g_population; p++) {
            int idx = p * visit_node_max;
            for (int n = 0; n < num_indiv_rtl; n++) {
                int rtl_idx = rand() % g_numRetailNodes;
                rtl_idx += g_retailNodeIndexBegin;
                visit_node_prsn[idx] = rtl_idx;
                idx++;
            }
            return 0;
        }
    }

}