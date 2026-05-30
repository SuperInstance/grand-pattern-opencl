CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra -std=c11
# OpenCL headers: try pkg-config, fall back to common paths
OPENCL_CFLAGS ?= $(shell pkg-config --cflags OpenCL 2>/dev/null)
OPENCL_LDFLAGS ?= $(shell pkg-config --libs OpenCL 2>/dev/null || echo "-lOpenCL")

SRCDIR = src
TESTDIR = tests
BUILDDIR = build

SRCS = $(SRCDIR)/host/opencl_host.c
TEST_SRC = $(TESTDIR)/test_harness.c

LIB = $(BUILDDIR)/libgrandpattern.a
TEST_BIN = $(BUILDDIR)/test_harness

.PHONY: all clean test

all: $(LIB)

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

$(BUILDDIR)/opencl_host.o: $(SRCDIR)/host/opencl_host.c $(SRCDIR)/host/opencl_host.h | $(BUILDDIR)
	$(CC) $(CFLAGS) $(OPENCL_CFLAGS) -I$(SRCDIR)/host -c -o $@ $<

$(LIB): $(BUILDDIR)/opencl_host.o
	ar rcs $@ $<

$(BUILDDIR)/test_harness.o: $(TESTDIR)/test_harness.c $(SRCDIR)/host/opencl_host.h | $(BUILDDIR)
	$(CC) $(CFLAGS) $(OPENCL_CFLAGS) -I. -I$(SRCDIR)/host -c -o $@ $<

$(TEST_BIN): $(BUILDDIR)/test_harness.o $(LIB)
	$(CC) $(CFLAGS) -o $@ $< -L$(BUILDDIR) -lgrandpattern $(OPENCL_LDFLAGS) -lm

test: $(TEST_BIN)
	./$(TEST_BIN)

clean:
	rm -rf $(BUILDDIR)
