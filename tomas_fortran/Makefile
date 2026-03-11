# Makefile for TOMAS Fortran (self-contained)
#
# Build the original TOMAS Fortran source from this directory.
# No external paths required — all source is in src/, src_ppm/, include/, harness/.

FC = gfortran
FFLAGS = -O0 -fdefault-real-8 -fdefault-double-8 -ffixed-form -std=legacy -Iinclude

# Directories
SRC = src
SRC_PPM = src_ppm
INC = include
HARNESS = harness

# Core TOMAS source files (TFL)
TOMAS_SRCS = $(SRC)/multicoag.f \
             $(SRC)/initbounds.f \
             $(SRC)/mnfix.f \
             $(SRC)/aerodens.f \
             $(SRC)/loginit.f \
             $(SRC)/ezcond.f \
             $(SRC)/tmcond.f \
             $(SRC)/dmdt_int.f \
             $(SRC)/getCondSink.f \
             $(SRC)/gasdiff.f \
             $(SRC)/eznh3eqm.f \
             $(SRC)/ezwatereqm.f \
             $(SRC)/waterso4.f \
             $(SRC)/waternacl.f

# Object files (built in current directory)
TOMAS_OBJS = multicoag.o initbounds.o mnfix.o aerodens.o loginit.o \
             ezcond.o tmcond.o dmdt_int.o getCondSink.o gasdiff.o \
             eznh3eqm.o ezwatereqm.o waterso4.o waternacl.o

# PPM source files (new)
PPM_SRCS = $(SRC_PPM)/tmcond_ppm.f \
           $(SRC_PPM)/ezcond_ppm.f

# PPM object files
PPM_OBJS = tmcond_ppm.o ezcond_ppm.o

# Shared objects needed by PPM (reuses TFL core minus ezcond.o/tmcond.o)
SHARED_OBJS = multicoag.o initbounds.o mnfix.o aerodens.o loginit.o \
              dmdt_int.o getCondSink.o gasdiff.o \
              eznh3eqm.o ezwatereqm.o waterso4.o waternacl.o

# Nucleation source files
NUC_SRCS = $(SRC)/ricco_nucl.f \
           $(SRC)/dunne_inorg_nucl.f

# Nucleation object files
NUC_OBJS = ricco_nucl.o dunne_inorg_nucl.o

# Executables
TARGET = benchmark.exe
TARGET_24H = benchmark_24h.exe
TARGET_24H_PPM = benchmark_24h_ppm.exe
TARGET_NUC = benchmark_nucleation.exe
TARGET_CONSTGC = benchmark_constgc.exe

# Default target
all: setup $(TARGET)

# Build 24-hour benchmark (TFL)
benchmark_24h: setup_24h $(TARGET_24H)

# Build 24-hour PPM benchmark
benchmark_24h_ppm: setup_24h $(TARGET_24H_PPM)

# Build nucleation benchmark (standalone, no sizecode.COM)
benchmark_nucleation: setup $(TARGET_NUC)

# Build constant-gas benchmark
benchmark_constgc: setup_constgc $(TARGET_CONSTGC)

# Build both TFL and PPM benchmarks
benchmark_both: benchmark_24h benchmark_24h_ppm

# Create output directories
setup:
	mkdir -p output

setup_24h:
	mkdir -p output/24h

setup_constgc:
	mkdir -p output/constgc

# Link
$(TARGET): benchmark_harness.o $(TOMAS_OBJS)
	$(FC) $(FFLAGS) -o $@ $^

$(TARGET_24H): benchmark_24h.o $(TOMAS_OBJS) $(NUC_OBJS) nucleation_driver.o
	$(FC) $(FFLAGS) -o $@ $^

$(TARGET_24H_PPM): benchmark_24h_ppm.o $(PPM_OBJS) $(SHARED_OBJS)
	$(FC) $(FFLAGS) -o $@ $^

$(TARGET_NUC): benchmark_nucleation.o $(NUC_OBJS)
	$(FC) $(FFLAGS) -o $@ $^

$(TARGET_CONSTGC): benchmark_constgc.o $(TOMAS_OBJS) $(NUC_OBJS) nucleation_driver.o
	$(FC) $(FFLAGS) -o $@ $^

# Compile harness files
benchmark_harness.o: $(HARNESS)/benchmark_harness.f $(INC)/sizecode.COM
	$(FC) $(FFLAGS) -c $<

benchmark_24h.o: $(HARNESS)/benchmark_24h.f $(INC)/sizecode.COM
	$(FC) $(FFLAGS) -c $<

benchmark_24h_ppm.o: $(HARNESS)/benchmark_24h_ppm.f $(INC)/sizecode.COM
	$(FC) $(FFLAGS) -c $<

benchmark_nucleation.o: $(HARNESS)/benchmark_nucleation.f
	$(FC) $(FFLAGS) -c $<

benchmark_constgc.o: $(HARNESS)/benchmark_constgc.f $(INC)/sizecode.COM
	$(FC) $(FFLAGS) -c $<

# Compile TOMAS sources
multicoag.o: $(SRC)/multicoag.f $(INC)/sizecode.COM
	$(FC) $(FFLAGS) -c $<

initbounds.o: $(SRC)/initbounds.f $(INC)/sizecode.COM
	$(FC) $(FFLAGS) -c $<

mnfix.o: $(SRC)/mnfix.f $(INC)/sizecode.COM
	$(FC) $(FFLAGS) -c $<

aerodens.o: $(SRC)/aerodens.f $(INC)/sizecode.COM
	$(FC) $(FFLAGS) -c $<

loginit.o: $(SRC)/loginit.f $(INC)/sizecode.COM
	$(FC) $(FFLAGS) -c $<

ezcond.o: $(SRC)/ezcond.f $(INC)/sizecode.COM
	$(FC) $(FFLAGS) -c $<

tmcond.o: $(SRC)/tmcond.f $(INC)/sizecode.COM
	$(FC) $(FFLAGS) -c $<

dmdt_int.o: $(SRC)/dmdt_int.f $(INC)/sizecode.COM
	$(FC) $(FFLAGS) -c $<

getCondSink.o: $(SRC)/getCondSink.f $(INC)/sizecode.COM
	$(FC) $(FFLAGS) -c $<

gasdiff.o: $(SRC)/gasdiff.f $(INC)/sizecode.COM
	$(FC) $(FFLAGS) -c $<

eznh3eqm.o: $(SRC)/eznh3eqm.f $(INC)/sizecode.COM
	$(FC) $(FFLAGS) -c $<

ezwatereqm.o: $(SRC)/ezwatereqm.f $(INC)/sizecode.COM
	$(FC) $(FFLAGS) -c $<

waterso4.o: $(SRC)/waterso4.f $(INC)/sizecode.COM
	$(FC) $(FFLAGS) -c $<

waternacl.o: $(SRC)/waternacl.f $(INC)/sizecode.COM
	$(FC) $(FFLAGS) -c $<

# Compile nucleation sources (standalone, no sizecode.COM)
ricco_nucl.o: $(SRC)/ricco_nucl.f
	$(FC) $(FFLAGS) -c $<

dunne_inorg_nucl.o: $(SRC)/dunne_inorg_nucl.f
	$(FC) $(FFLAGS) -c $<

nucleation_driver.o: $(SRC)/nucleation_driver.f $(INC)/sizecode.COM
	$(FC) $(FFLAGS) -c $<

# Compile PPM sources
tmcond_ppm.o: $(SRC_PPM)/tmcond_ppm.f $(INC)/sizecode.COM
	$(FC) $(FFLAGS) -c $<

ezcond_ppm.o: $(SRC_PPM)/ezcond_ppm.f $(INC)/sizecode.COM
	$(FC) $(FFLAGS) -c $<

# Run targets
run: $(TARGET)
	./$(TARGET)

run_24h: $(TARGET_24H)
	./$(TARGET_24H)

run_24h_ppm: $(TARGET_24H_PPM)
	./$(TARGET_24H_PPM)

# Clean
clean:
	rm -f *.o $(TARGET) $(TARGET_24H) $(TARGET_24H_PPM) $(TARGET_NUC) $(TARGET_CONSTGC)
	rm -rf output

run_constgc: $(TARGET_CONSTGC)
	./$(TARGET_CONSTGC)

.PHONY: all setup setup_24h setup_constgc run run_24h run_24h_ppm run_constgc clean \
        benchmark_24h benchmark_24h_ppm benchmark_both benchmark_nucleation benchmark_constgc
