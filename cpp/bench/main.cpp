// Microbenchmarks of the C++ core, without Python overhead. Run with cpp/bench/run.sh.
void bench_kepler();
void bench_two_body();

int main() {
  bench_kepler();
  bench_two_body();
  return 0;
}
