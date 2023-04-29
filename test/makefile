nguyen:
	python3 $(target) --task=Nguyen-09 &
	python3 $(target) --task=Nguyen-10 &
	python3 $(target) --task=Nguyen-11 &
	python3 $(target) --task=Nguyen-12 &

jin:
	python3 $(target) --task=Jin-1 &
	python3 $(target) --task=Jin-2 &
	python3 $(target) --task=Jin-3 &
	python3 $(target) --task=Jin-4 &
	python3 $(target) --task=Jin-5 &
	python3 $(target) --task=Jin-6 &

nggp-jin:
	python3 run_nggp.py --task=Jin-1 > log_nggp_1 &
	python3 run_nggp.py --task=Jin-2 > log_nggp_2 &
	python3 run_nggp.py --task=Jin-3 > log_nggp_3 &
	python3 run_nggp.py --task=Jin-4 > log_nggp_4 &
	python3 run_nggp.py --task=Jin-5 > log_nggp_5 &
	python3 run_nggp.py --task=Jin-6 > log_nggp_6 &

dsr-jin:
	python3 run_dsr.py --task=Jin-1 > log_dsr_1 &
	python3 run_dsr.py --task=Jin-2 > log_dsr_2 &
	python3 run_dsr.py --task=Jin-3 > log_dsr_3 &
	python3 run_dsr.py --task=Jin-4 > log_dsr_4 &
	python3 run_dsr.py --task=Jin-5 > log_dsr_5 &
	python3 run_dsr.py --task=Jin-6 > log_dsr_6 &

show:
	ps aux | grep $(name) | grep -v grep

kill: 
	ps aux | grep $(name) | grep -v grep | cut -c 8-17 | xargs kill -9