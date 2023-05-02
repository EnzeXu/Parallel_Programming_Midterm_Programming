nguyen:
	python3 $(target) --task=Nguyen-09 > log_nguyen_$(target)_09 &
	python3 $(target) --task=Nguyen-10 > log_nguyen_$(target)_10 &
	python3 $(target) --task=Nguyen-11 > log_nguyen_$(target)_11 &
	python3 $(target) --task=Nguyen-12 > log_nguyen_$(target)_12 &

jin:
	python3 $(target) --task=Jin-1 > log_jin_$(target)_1 &
	python3 $(target) --task=Jin-2 > log_jin_$(target)_2 &
	python3 $(target) --task=Jin-3 > log_jin_$(target)_3 &
	python3 $(target) --task=Jin-4 > log_jin_$(target)_4 &
	python3 $(target) --task=Jin-5 > log_jin_$(target)_5 &
	python3 $(target) --task=Jin-6 > log_jin_$(target)_6 &

korns:
	python3 $(target) --task=Korns-2  > log_Korns_$(target)_2  &
	python3 $(target) --task=Korns-3  > log_Korns_$(target)_3  &
	python3 $(target) --task=Korns-8  > log_Korns_$(target)_8  &
	python3 $(target) --task=Korns-9  > log_Korns_$(target)_9  &
	python3 $(target) --task=Korns-10 > log_Korns_$(target)_10 &
	python3 $(target) --task=Korns-11 > log_Korns_$(target)_11 &
	python3 $(target) --task=Korns-12 > log_Korns_$(target)_12 &
	python3 $(target) --task=Korns-13 > log_Korns_$(target)_13 &
	python3 $(target) --task=Korns-14 > log_Korns_$(target)_14 &
	python3 $(target) --task=Korns-15 > log_Korns_$(target)_15 &

show:
	ps aux | grep $(name) | grep -v grep

kill: 
	ps aux | grep $(name) | grep -v grep | cut -c 8-17 | xargs kill -9