# Makefile for TOMAS Coagulation Benchmark Harness
#
# Compile with -O0 (no optimization) for reproducibility.
# Uses gfortran by default.

FC = gfortran
FFLAGS = -O0 -fdefault-real-8 -fdefault-double-8 -ffixed-form -std=legacy -I.

# Path to original TOMAS source files
# From benchmarks/fortran/ go up 4 levels to aerosol-microphysics/
TOMAS_SRC = ../../../../original-models/TOMAS/src
TOMAS_INC = ../../../../original-models/TOMAS/inputs/created_files

# Source files
HARNESS = benchmark_harness.f
TOMAS_SRCS = $(TOMAS_SRC)/multicoag.f \
             $(TOMAS_SRC)/initbounds.f \
             $(TOMAS_SRC)/mnfix.f \
             $(TOMAS_SRC)/aerodens.f \
             $(TOMAS_SRC)/loginit.f \
             $(TOMAS_SRC)/ezcond.f \
             $(TOMAS_SRC)/tmcond.f \
             $(TOMAS_SRC)/dmdt_int.f \
             $(TOMAS_SRC)/getCondSink.f \
             $(TOMAS_SRC)/gasdiff.f \
             $(TOMAS_SRC)/eznh3eqm.f \
             $(TOMAS_SRC)/ezwatereqm.f \
             $(TOMAS_SRC)/waterso4.f \
             $(TOMAS_SRC)/waternacl.f

# Object files
HARNESS_OBJ = benchmark_harness.o
TOMAS_OBJS = multicoag.o initbounds.o mnfix.o aerodens.o loginit.o \
             ezcond.o tmcond.o dmdt_int.o getCondSink.o gasdiff.o \
             eznh3eqm.o ezwatereqm.o waterso4.o waternacl.o

# Executables
TARGET = benchmark.exe
TARGET_24H = benchmark_24h.exe

# Default target
all: setup $(TARGET)

# Build 24-hour benchmark
benchmark_24h: setup_24h $(TARGET_24H)

# Create output directories
setup:
	mkdir -p output

setup_24h:
	mkdir -p output/24h

# Link
$(TARGET): $(HARNESS_OBJ) $(TOMAS_OBJS)
	$(FC) $(FFLAGS) -o $@ $^

# Link 24h benchmark
$(TARGET_24H): benchmark_24h.o $(TOMAS_OBJS)
	$(FC) $(FFLAGS) -o $@ $^

# Compile harness (needs sizecode.COM)
benchmark_harness.o: $(HARNESS) sizecode.COM
	$(FC) $(FFLAGS) -c $<

# Compile 24h harness
benchmark_24h.o: benchmark_24h.f sizecode.COM
	$(FC) $(FFLAGS) -c $<

# Compile TOMAS sources (need sizecode.COM)
multicoag.o: $(TOMAS_SRC)/multicoag.f sizecode.COM
	$(FC) $(FFLAGS) -c $<

initbounds.o: $(TOMAS_SRC)/initbounds.f sizecode.COM
	$(FC) $(FFLAGS) -c $<

mnfix.o: $(TOMAS_SRC)/mnfix.f sizecode.COM
	$(FC) $(FFLAGS) -c $<

aerodens.o: $(TOMAS_SRC)/aerodens.f sizecode.COM
	$(FC) $(FFLAGS) -c $<

loginit.o: $(TOMAS_SRC)/loginit.f sizecode.COM
	$(FC) $(FFLAGS) -c $<

ezcond.o: $(TOMAS_SRC)/ezcond.f sizecode.COM
	$(FC) $(FFLAGS) -c $<

tmcond.o: $(TOMAS_SRC)/tmcond.f sizecode.COM
	$(FC) $(FFLAGS) -c $<

dmdt_int.o: $(TOMAS_SRC)/dmdt_int.f sizecode.COM
	$(FC) $(FFLAGS) -c $<

getCondSink.o: $(TOMAS_SRC)/getCondSink.f sizecode.COM
	$(FC) $(FFLAGS) -c $<

gasdiff.o: $(TOMAS_SRC)/gasdiff.f sizecode.COM
	$(FC) $(FFLAGS) -c $<

eznh3eqm.o: $(TOMAS_SRC)/eznh3eqm.f sizecode.COM
	$(FC) $(FFLAGS) -c $<

ezwatereqm.o: $(TOMAS_SRC)/ezwatereqm.f sizecode.COM
	$(FC) $(FFLAGS) -c $<

waterso4.o: $(TOMAS_SRC)/waterso4.f sizecode.COM
	$(FC) $(FFLAGS) -c $<

waternacl.o: $(TOMAS_SRC)/waternacl.f sizecode.COM
	$(FC) $(FFLAGS) -c $<

# Symlink sizecode.COM from TOMAS includes
sizecode.COM:
	ln -sf $(TOMAS_INC)/sizecode.COM .

# Run benchmark
run: $(TARGET)
	./$(TARGET)

# Clean
clean:
	rm -f *.o $(TARGET) $(TARGET_24H) sizecode.COM
	rm -rf output

# Run 24h benchmark
run_24h: $(TARGET_24H)
	./$(TARGET_24H)

.PHONY: all setup setup_24h run run_24h clean benchmark_24h
