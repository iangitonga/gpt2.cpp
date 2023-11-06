#pragma once

#include <memory>
#include <vector>
#include <cstdio>
#include <cstdlib>

#include "gten_types.h"


// Assert that the given boolean is true. If false, print message and terminate program.
// TODO: Replace with C++ 20 __VA_OPT__, __VA_ARGS__ may not work on non-gcc compilers.
#define GTEN_ASSERT(condition, message, ...)                                              \
    if (__glibc_unlikely(!(condition))) {                                                                   \
        std::fprintf(stderr, "\x1B[1;31m");                                             \
        std::fprintf(stderr, "\nGTEN ERROR [File `%s` line %d]: ", __FILE__, __LINE__);   \
        std::fprintf(stderr, message, ##__VA_ARGS__);                                   \
        std::fprintf(stderr, "\n");                                                     \
        std::exit(EXIT_FAILURE);                                                        \
    }


namespace gten {

// class TensorDims {
//     TensorDims(std::initializer_list<int> dims) {
//         GTEN_ASSERT(dims.size() <= 3, "Expected ndims <=3 but got ndims=%ld.", dims.size());
//         for (int i = 0; i < dims.size(); i++) {
//             const int dim = *(dims.begin() + i);
//             GTEN_ASSERT(dim > 0, "The value of dimension %d: %d of the given shape is invalid!", i, dim);
//             shape_[i] = dim;
//         }
//     }
//     friend std::ostream& operator<<(std::ostream& stream, TensorDims dims);

// private:
//     int shape_[3];
//     int strides_[3];
//     int dimsize;
// };

static int64_t G_TensorMemAllocated = 0;


class Tensor {
public:
    Tensor() = default;
    Tensor(const std::vector<int>& shape, TensorDtype dtype, float qscale = 0, int qzerop = 0);
    Tensor(void* data_ptr, const std::vector<int>& shape, TensorDtype dtype, float qscale = 0, int qzerop = 0);
    Tensor(const Tensor& rhs) = default;
    Tensor(Tensor&& rhs) = default;
    Tensor& operator=(const Tensor& rhs) = default;
    Tensor &operator=(Tensor&& rhs) = default;

    // Get the pointer to internal data buffer.
    template <typename T>
    T* data_ptr() { 
        return reinterpret_cast<T*>(data_ptr_.get()); 
    }

    template <typename T>
    const T* data_ptr() const { 
        return reinterpret_cast<const T*>(data_ptr_.get()); 
    }

    TensorDtype dtype() const {
        return dtype_;
    }

    // Get the number of bytes that an element in the tensor occupies.
    int itemsize() const {
        switch (dtype_) {
            case kQint8:
                return 1;
            case kInt32:
                return 4;
            case kFloat16:
                return 2;
            case kFloat32:
                return 4;
            default:
                std::cout << "defaulting on itemsize\n";
                return 4;
        }
    }

    int ndims() const {
        return shape_.size();
    }

    // Get the number of elems in the tensor.
    int numel() const {
        return numel_;
    }

    /// Returns the size of the give dimension.
    int size(int i) const {
        GTEN_ASSERT(i < int(shape_.size()),
                    "The given index `%d` is out of range of shape with size `%d`.",
                    i, int(shape_.size()));
        return shape_[i];
    }

    /// Returns the size of the give dimension.
    int stride(int i) const {
        GTEN_ASSERT(i < int(strides_.size()),
                    "The given index `%d` is out of range of stride with size `%d`.",
                    i, int(strides_.size()));
        return strides_[i];
    }

    size_t nbytes() const {
        return numel_ * itemsize();
    }

    const std::vector<int>& shape() const {
        return shape_;
    }

    bool shape_eq(const std::vector<int>& shape) const {
        return shape == shape_;
    }

    float scale() const {
        return qscale_;
    }

    int zerop() const {
        return qzerop_;
    }

    void set_qparams(float scale, int zerop) {
        GTEN_ASSERT(dtype_ == kQint8, "Qparams can only be set for dtype=Qint8, not %s.", dtype_str(dtype_));
        GTEN_ASSERT(scale != 0, "Expected non-zero scale for dtype Qint8 but got %f.", scale);
        qscale_ = scale;
        qzerop_ = zerop;
    }

    // Resize the tensor to have a new shape. The new shape must not be larger than the
    // shape provided when the tensor was created because this function does not
    // reallocate tensor storage.
    // Note: this is not a reshape function because a reshape function can only reshape
    // a tensor if the new and the existing shapes have the same number of elements.
    void resize(const std::vector<int>& new_shape);

    friend std::ostream& operator<<(std::ostream& stream, const Tensor& tensor);
    void print() const;
    void print_info() const;

    Tensor view(const std::vector<int>& new_shape) const;
    Tensor permute(const std::vector<int>& new_shape);
    void set_strides(const std::vector<int>& strides);
    std::string shape_str() const;
    std::string strides_str() const;

private:
    TensorDtype dtype_ = kFloat32;
    std::shared_ptr<uint8_t[]> data_ptr_;
    int storage_size_ = 0;  // in_bytes
    int numel_ = 0;
    std::vector<int> shape_;
    std::vector<int> strides_;
    float qscale_ = 0.0f;
    int qzerop_ = 0;

    void validate_shape(const std::vector<int>& shape) const;
    void set_strides_from_shape(const std::vector<int>& shape);
    int numel_from_shape(const std::vector<int>& shape) const;
    void print_single(int item_idx, int col_idx, int n_cols) const;
};

} // Namespace xten
