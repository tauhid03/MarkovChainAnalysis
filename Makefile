
PYTHONPATH="$$PYTHONPATH":$(CURDIR)

export PYTHONPATH

exp1: exp1-n2.out exp1-n3.out \
	  exp1-n4.out exp1-n5.out \
	  exp1-n6.out exp1-n7.out \
	  exp1-n8.out exp1-n9.out

exp1-n%.out:
	python3 examples/multiagent_closed_loop_control.py $* &> $@
