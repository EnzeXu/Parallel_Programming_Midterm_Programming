run:
	python3 ../src/main.py $(test)

rerun:
	rm -rf results/$(test)
	python3 ../src/main.py $(test)

rerun-mksr:
	rm -rf results/$(test)/mksr_y0
	rm -rf results/$(test)/mksr_y1
	python3 ../src/main.py $(test)

clean:
	rm -rf results
	