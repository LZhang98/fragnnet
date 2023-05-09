import argparse
import concurrent.futures
import logging
import multiprocessing
import os
import subprocess
import time
import signal
from threading import Event
import numpy as np
import time
from pprint import pprint
import argparse
import torch_geometric as pyg

import frag_gnn.frag.frag_utils as frag_utils
from frag_gnn.utils.misc_utils import timeit, booltype, get_pyg_memory_usage

from typing import Any, Tuple, Dict
from collections import deque

class ElapsedFormatter(logging.Formatter):
    def __init__(self):
        super().__init__()
        self.start_time = time.time()

    def format(self, record):
        # using timedelta here for convenient default formatting
        time_str = self.get_time_between(record.created)
        return "[{}] [{}] {}".format(time_str, record.levelname, record.getMessage())

    def get_time_between(self, current):
        hours, rem = divmod(current - self.start_time, 3600)
        minutes, seconds = divmod(rem, 60)
        return "{:0>2}:{:0>2}:{:0>2}".format(int(hours), int(minutes), int(seconds))


class FragTaskManager(object):

    def __init__(self, tasks:list = None) -> None:
        # setup signal handler
        self.exit_event = Event()
        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)
        signal.signal(signal.SIGUSR1, self.handle_signal)
        # setup task queue
        if tasks is not None:
            self.task_q = deque(tasks)
        else:
            self.task_q = deque()

        self._latest_tast_id = 0
        self._job_queue_size = 16

    def handle_signal(self, signal_value, _):
        """Methods handle signal, right now it only handle exit
        Args:
            signal_value (_type_): _description_
            _ (_type_): _description_
        """
        logger = logging.getLogger('frag_batch')
        signame = signal.Signals(signal_value).name
        logger.info('Get signal {}'.format(signame))
        self.exit_event.set()

    def add_tasks(self, futures_in_working, executor):
        """method add task to executor

        Args:
            futures_in_working (_type_): _description_
            executor (_type_): _description_
        """
        logger = logging.getLogger('frag_batch')
        #tasks_info = self.get_tasks()

        # populate data
        #for task_info in tasks_info:
        allowed_new_task = self._job_queue_size - 
            smiles, max_depth, h_prior, max_h_transfer, frag_max_time  = task_info
            futures_in_working[executor.submit(
                FragTaskManager.run_frag_task, smiles, smiles, max_depth, h_prior, max_h_transfer, frag_max_time)] =  self._latest_tast_id
            self._latest_tast_id += 1
        logger.info('Added {} tasks'.format(len(tasks_info)))

    
    def run_jobs(self, num_processes, sleep_interval, output_dir):

        logger = logging.getLogger('frag_batch')
        finished_task = 0
        if self.exit_event.is_set():
            return
        
        # We can use a with statement to ensure threads are cleaned up promptly
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_processes) as executor:
            # start a future for a thread which sends work in through the queue
            # futures_in_working = {executor.submit(get_tasks, db_file, worker_grp_id) : ""}

            futures_in_working = {}
            self.add_tasks(futures_in_working, executor)

            while futures_in_working and not self.exit_event.is_set():
                # check for status of the futures which are currently working
                completed, not_completed = concurrent.futures.wait(
                    futures_in_working, timeout=0.25,
                    return_when=concurrent.futures.FIRST_COMPLETED)

                # process any completed futures
                finished_tasks = []
                completed_this_interval = len(completed)
                predicted_spectra_this_interval = 0
                for future in completed:
                    try:
                        data = future.result()
                    except Exception as exc:
                        print('Generated an exception:', exc)
                    else:
                        finished_task_id = futures_in_working[future]
                        #ionization_type, task_stats, predicted_spectra_str = data
                        finished_tasks.append(finished_task_id)
                        finished_task += 1
                        
                    # remove the now completed future
                    del futures_in_working[future]

                # Update database
                # if there is incoming work, and there still space in the query start a new future
                num_not_completed = len(not_completed)
                
                logger.info('Finished {} prediction tasks, predicted {} spectra set, in last {} second'.format(completed_this_interval, predicted_spectra_this_interval, sleep_interval))
                logger.info('{} tasks in the queue. Finished {} prediction tasks since start.'.format(num_not_completed, finished_task))
                logger.info('Task manager idle for {:.2f} s.'.format(sleep_interval))
                predicted_spectra_this_interval = 0
                if num_not_completed:
                    self.exit_event.wait(sleep_interval)
                #time.sleep(sleep_interval)

            if self.exit_event.is_set():
                #Force kill, this is a dirty but quick way to exit
                logger.info('Shuting down signal received.')
                for future in futures_in_working:
                    future.cancel()
                logger.info('Terminating child processes.')
                # Run this 3 times to make sure all child processes are killed
                # Because there are two sub process call in prediction tasks, one kill command will not kill all of them in some cases
                for idx in range(3):
                    logger.info('kill child processes, part {}.'.format(idx+1))
                    task_proc = subprocess.Popen(['pgrep -P {} | xargs kill'.format(str(os.getpid()))], stdout=subprocess.PIPE, shell=True)
                    _, _ = task_proc.communicate()
                    time.sleep(1)    
                logger.info('Shuting down executor.')
                executor.shutdown(wait=True)
                logger.info('Shuting down completed.')
            
    @staticmethod
    def run_frag_task(smiles, max_depth, h_prior, max_h_transfer, frag_max_time):
        mol_d = frag_utils.extract_mol_info(smiles)
        num_nodes = mol_d["atom_mask_arr"].shape[0]
        print(num_nodes)

        dag_d = timeit(frag_utils.compute_dags)(mol_d,max_depth,h_prior,max_h_transfer,frag_max_time)
        dag = dag_d["dag"]
        print(type(dag))
        print("> dag")
        dag_nn = dag.num_nodes
        dag_ne = dag.num_edges
        dag_indeg = pyg.utils.degree(dag.edge_index[1]).numpy()
        print(f"num_nodes = {dag_nn}, num_edges = {dag_ne}, edge_frac = {2*dag_ne/(dag_nn*(dag_nn-1))}")
        print(f"avg_indeg = {np.mean(dag_indeg)}, std_indeg = {np.std(dag_indeg)}")
        dag_memory = get_pyg_memory_usage(dag)
        print(f"memory = {dag_memory/1e6} MB")
        return dag_d
  
if __name__ == '__main__':

    # we could make it global ... but I just don't like global variables
    term_event = Event()
    def main_signal_hander():
        term_event.set()

    signal.signal(signal.SIGINT, main_signal_hander)
    signal.signal(signal.SIGTERM, main_signal_hander)
    signal.signal(signal.SIGUSR1, main_signal_hander)

    # get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--sleep_interval',
                        help='init sleep interval', type=int, default=30)    
    parser.add_argument('--output_dir', help='output dir', type=str, default='./frag_dags')

    args = parser.parse_args()
    # get time
    start_time = time.time()

    # start or numbewr of process
    num_processes = multiprocessing.cpu_count()

    # build output folders for output file
    output_dir = args.output_dir
    dir_path = os.path.dirname(output_dir)
    os.makedirs(dir_path, exist_ok= True)

    # Main thread sleep interval
    # place holder here
    sleep_interval = args.sleep_interval

    log_level = logging.DEBUG
    # setup logger
    logger = logging.getLogger('frag_batch')
    logger.setLevel(log_level)

    # create console handler
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    # create formatter and add it to the handlers
    formatter = ElapsedFormatter()
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(ch)

   
    # logging some info
    logger.info('=' * 120)
    logger.info('cwd: {}'.format(os.getcwd()))

    logger.info('Sleep interval: {}s'.format(sleep_interval))
    logger.info('Num of processors: {}'.format(num_processes))
    #logger.info('output_include_id: {}'.format(output_include_id))

    IDX_TO_SMILES = {
		0: "CCO", # ethanol
		1: "c1ccccc1", # benzene
		2: "CC(=O)OC1=CC=CC=C1C(=O)O", # aspirin
		3: "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C", # testosterone
		4: "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC(=O)O", # geddic acid
		5: "CSCCC(C(=O)O)N", # methionine, S=2
		6: "C1=CC(=CC=C1N)S(=O)(=O)N", # sulfanilamide, S=6
		7: "COP(C1=CC=CC=C1)C2=CC=CC=C2", # methoxydiphenylphosphine, P=3
		8: "CC1C(O1)P(=O)(O)O", # phosphomycin, P=5
		9: 'CC(=O)NC1C(OC2CCC3(C)C(CCC4(C)C3CC=C3C5CC(C)(C)CCC5(C(=O)O)CCC34C)C2(C)C)OC(CO)C(O)C1O', # big one (47 atoms)
		10: 'CC(=O)NC1C(OC2OC(CC(O)C3OC(n4ccc(=O)[nH]c4=O)C(O)C3O)C(O)C(O)C2NC(=O)C=CCCCCCCCCCCCC(C)C)OC(CO)C(O)C1O', # big one (60 atoms)
		11: 'C12=C3C4=C5C6=C3C7=C1C8=C9C2=C4C1=C5C2=C6C7=C8C2=C91', #Fullerene-C20
		12: 'c12c3c4c5c1c1c6c7c2c2c8c3c3c9c4c4c%10c5c5c1c1c6c6c%11c7c2c2c7c8c3c3c8c9c4c4c9c%10c5c5c1c1c6c6c%11c2c2c7c3c3c8c4c4c9c5c1c1c6c2c3c41', #	C60 fullerene
		13: 'CC(=O)OC12COC1CC(O)C1(C)C(=O)C(=O)C3C(C)C(OC(=O)C(O)C(NC(=O)c4ccccc4)c4ccccc4)CC(O)(C(OC(=O)c4ccccc4)C21)C3(C)C'	
	}

    max_depth = 4
    h_prior = True
    max_h_transfer = 4
    frag_max_time = None
    smiles = IDX_TO_SMILES
    tasks = [[smiles, max_depth, h_prior, max_h_transfer, frag_max_time] for smiles in IDX_TO_SMILES]

    if not term_event.is_set():
        task_mgt = FragTaskManager(tasks)
        task_mgt.run_jobs(num_processes, sleep_interval, output_dir)