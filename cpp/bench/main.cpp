// Microbenchmarks of the C++ core, without Python overhead. Run with cpp/bench/run.sh.
void bench_kepler();
void bench_two_body();
void bench_two_body_batched();

int main() {
  bench_kepler();
  bench_two_body();
  bench_two_body_batched();
  return 0;
}
