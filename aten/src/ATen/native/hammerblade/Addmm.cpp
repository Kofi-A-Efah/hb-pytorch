#include <ATen/ATen.h>
#include <ATen/hammerblade/HammerBladeContext.h>
#include <ATen/native/hammerblade/Offload.h>

namespace at { namespace native {

Tensor addmm_hb(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    Scalar beta,
    Scalar alpha
) {

  //TORCH_CHECK(false, "addmm_hb not implemented");
  //TODO: implmement the correct addmm_hb host code
  //      you may use void offload_tensor_scalar_impl(std::vector<Tensor>, std::vector<Scalar>,
  //                                                  cosn char*)
  //      which is defined in Offload.h for kernel offloading

  if (self.dim() != mat1.dim() || self.dim() != mat2.dim() || mat1.dim() != mat2.dim()) {
    AT_ERROR("All matrices must be of the same dimension; got ", self.dim(), "D, ", mat1.dim(), "D, ", mat2.dim(), "D tensors");
  }

  if ( (self.scalar_type() != ScalarType::Float)
    || (mat1.scalar_type() != ScalarType::Float)
    || (mat2.scalar_type() != ScalarType::Float) ) {
    AT_ERROR("HammerBlade addmm is implemented for Float only");
  }

  TORCH_CHECK(mat1.size(1) == mat2.size(0), "Argument #3: Expected dim 0 size ", mat1.size(1), ", got ", mat2.size(0));

  Tensor b_res;
  std::tie(b_res) = expand_size(self, {mat1.size(0), mat2.size(1)}, "addmm_out"); 
  Tensor result = at::empty({}, b_res.options());
  offload_tensor_scalar_impl(result, self, mat1, mat2, beta, alpha, "tensorlib_addmm");

  return result;  
}

}} // namespace at::native
