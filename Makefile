CC=g++-5

EXE=nn

CFLAGS=-std=c++11 -fopenmp -mavx2 -ftree-vectorize -Ofast -ffast-math  -g 
# 
SRCPATH=./src/cpp/
OBJPATH=./obj

#Variable setting the number of threads executing in parallel
export OMP_NUM_THREADS=4

objects = $(addprefix $(OBJPATH)/, activation.o timer.o paramsinit.o unit.o layer.o objective.o datareader.o neuralnetwork.o contractiveautoencoder.o autoencoder.o trainingalgorithm.o initialiser.o outwriter.o  main.o )

$(EXE): $(objects)
	$(CC) $(objects) -lconfig++ -fopenmp -o $@

$(OBJPATH) $(OBJPATH)/%.o: $(SRCPATH)%.cpp
	$(CC) -c $(CFLAGS) $< -o $@

run: $(EXE)
	./$(EXE)

plot:
	matlab -nosplash -nodisplay -r "run ./scripts/matlab/main.m, quit()"

.PHONY: clean
clean:
	rm $(EXE) $(objects)

