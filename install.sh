g++ -c -Ofast -fopenmp -fPIC Path/GA_CPP/GA/ga_bridge.cc -o Path/GA_CPP/GA/ga_bridge.o -ljsoncpp
g++ -shared -fopenmp -Wl,-soname,ga_bridge.so -o Path/GA_CPP/GA/ga_bridge.so Path/GA_CPP/GA/ga_bridge.o -ljsoncpp
rm -f Path/GA_CPP/GA/ga_bridge.o

g++ -c -Ofast -fopenmp -fPIC Path/GA_CPP/GAH/ga_h_bridge.cc -o Path/GA_CPP/GAH/ga_h_bridge.o -ljsoncpp
g++ -shared -fopenmp -Wl,-soname,ga_h_bridge.so -o Path/GA_CPP/GAH/ga_h_bridge.so Path/GA_CPP/GAH/ga_h_bridge.o -ljsoncpp
rm -f Path/GA_CPP/GAH/ga_h_bridge.o


g++ -c -Ofast -fopenmp -fPIC Arc/GA_CPP/GA/arc_ga_bridge.cc -o Arc/GA_CPP/GA/arc_ga_bridge.o -ljsoncpp
g++ -shared -fopenmp -Wl,-soname,Arc/GA_CPP/GA/arc_ga_bridge.so -o Arc/GA_CPP/GA/arc_ga_bridge.so Arc/GA_CPP/GA/arc_ga_bridge.o -ljsoncpp
rm -f Arc/GA_CPP/arc_ga_bridge.o

g++ -c -Ofast -fopenmp -fPIC Arc/GA_CPP/GAH/arc_ga_h_bridge.cc -o Arc/GA_CPP/GAH/arc_ga_h_bridge.o -ljsoncpp
g++ -shared -fopenmp -Wl,-soname,Arc/GA_CPP/GAH/arc_ga_h_bridge.so -o Arc/GA_CPP/GAH/arc_ga_h_bridge.so Arc/GA_CPP/GAH/arc_ga_h_bridge.o -ljsoncpp
rm -f Arc/GA_CPP/GAH/arc_ga_h_bridge.o