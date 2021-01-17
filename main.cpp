#include <fstream>
#include <sstream>
#include <stan/callbacks/interrupt.hpp>
#include <stan/callbacks/logger.hpp>
#include <stan/callbacks/stream_logger.hpp>
#include <stan/callbacks/stream_writer.hpp>
#include <stan/callbacks/writer.hpp>
#include <stan/io/empty_var_context.hpp>
#include <stan/services/sample/hmc_nuts_diag_e_adapt.hpp>

// This header contains model definition that doesn't use Stan-Math
#include "rosenbrock.hpp"

int main() {
  rosenbrock_model model(2);

  stan::callbacks::stream_writer info(std::cout);
  stan::callbacks::stream_writer err(std::cout);
  stan::callbacks::stream_logger logger(std::cout, std::cout, std::cout,
                                        std::cerr, std::cerr);
  stan::callbacks::writer init_writer, diagnostic_writer;
  stan::callbacks::interrupt interrupt;
  stan::io::empty_var_context context;

  std::fstream output_stream("output_rosenbrock.csv", std::fstream::out);
  stan::callbacks::stream_writer sample_writer(output_stream, "# ");

  sample_writer("model = " + model.model_name());

  unsigned int random_seed = 0;
  unsigned int id = 1;
  double init_radius = 0;
  int num_warmup = 1000;
  int num_samples = 10000;
  int num_thin = 1;
  bool save_warmup = true;
  int refresh = 100;
  double stepsize = 0.1;
  double stepsize_jitter = 0;
  int max_depth = 10;
  double delta = .8;
  double gamma = .05;
  double kappa = .75;
  double t0 = 10;
  unsigned int init_buffer = 75;
  unsigned int term_buffer = 50;
  unsigned int window = 25;

  int return_code = stan::services::sample::hmc_nuts_diag_e_adapt(
      model, context, random_seed, id, init_radius, num_warmup, num_samples,
      num_thin, save_warmup, refresh, stepsize, stepsize_jitter, max_depth,
      delta, gamma, kappa, t0, init_buffer, term_buffer, window, interrupt,
      logger, init_writer, sample_writer, diagnostic_writer);

  return return_code;
}