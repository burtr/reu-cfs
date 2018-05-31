


#PY= python3
PY= python
MaF= Makefile

HW= hello-world.py
FM= findmax.py
SS= sel-sort.py
SI= separated-int.py
MF= median-find.py
LAS= long-asc-subseq.py
BO= boom.py

hello:
	${PY} ${HW}

test-all:
	make -f ${MaF} test-boom
	${PY} ${FM}
	${PY} ${SS}
	${PY} ${SI}
	${PY} ${MF}
	${PY} ${LAS}

test-boom:
	${PY} ${BO} > boom.out 
	diff boom.ref boom.out
	-rm boom.out
