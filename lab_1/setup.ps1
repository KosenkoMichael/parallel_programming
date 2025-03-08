mkdir build
cd build
cmake .. -G "MinGW Makefiles"
cmake --build .
cd ..
python main.py