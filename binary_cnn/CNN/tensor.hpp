#define XTENSOR_USE_XSIMD
#include "xtensor/xarray.hpp"
#include "xtensor/xfixed.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xstrided_view.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xaccumulator.hpp"
#include "xtensor/xexpression.hpp"
#include "xtensor/xadapt.hpp"
#include "xtensor/xnoalias.hpp"
#include "xtensor-blas/xlinalg.hpp"
#include <chrono>  // for high_resolution_clock
#include "gradient_t.h"

using namespace xt;
using namespace std;
using namespace xt::placeholders;  // required for `_` to work

using Clock = std::chrono::high_resolution_clock;
using tensor_1d = xtensor<float, 1>;
using tensor_2d = xtensor<float, 2>;
using tensor_3d = xtensor<float, 3>;
using tensor_4d = xtensor<float, 4>;

using tensorg_1d = xtensor<gradient_t, 1>;
using tensorg_2d = xtensor<gradient_t, 2>;
using tensorg_3d = xtensor<gradient_t, 3>;
using tensorg_4d = xtensor<gradient_t, 4>;
