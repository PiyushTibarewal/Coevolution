clear
g++ -c lib/rng.cpp
g++ -o coevolution main.cpp rng.o
./coevolution -N 100 -T 100 -finSp 0 -ofn trace.txt -cfn cas.txt -mfn model.txt -wl 0 -mu 0.0001 -alpha 0.5 -eta 0.5 -beta 0.5 -sp 0.004 -w_phi 1 -w_kap 1 -rnd 0 2> log.txt
