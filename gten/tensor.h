#pragma once

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include "gten_types.h"

namespace gten
{

class Tensor {
public:
    Tensor() = default;

    // Construct a tensor with uninitialized data of the given shape and dtype.
    Tensor(std::initializer_list<int> shape, Dtype dtype);

    // Construct a tensor from an external data source. The constructed tensor does not take
    // ownership of the memory referenced by the pointer.
    Tensor(void* data_ptr, std::initializer_list<int> shape, Dtype dtype);
 
    // Copy-construct tensor from source tensor. Data from source tensor is shared with
    // the new tensor and thus data itself is not copied from source.
    Tensor(const Tensor& rhs) = default;

    // Move-construct a tensor from source tensor.
    Tensor(Tensor&& rhs) = default;

    // Copy-assign tensor from source tensor. Data from source tensor is shared with the
    // new tensor and thus data itself is not copied from source.
    Tensor &operator=(const Tensor& rhs) = default;

    // Move-assign a tensor from source tensor.
    Tensor &operator=(Tensor&& rhs) = default;
    
    // Get the pointer to internal data buffer.
    template <typename T>
    T* data_ptr() { 
        return reinterpret_cast<T*>(data_.get()); 
    }

    template <typename T>
    const T* data_ptr() const { 
        return reinterpret_cast<const T*>(data_.get()); 
    }

    Dtype dtype() const noexcept {
        return dtype_;
    }

    // Get the number of bytes that an element in the tensor occupies.
    size_t itemsize() const noexcept {
        return (dtype_ == kFloat16) ? 2 : 4;
    }

    int32_t ndims() const noexcept {
        return ndims_;
    }

    // Get the number of elems in the tensor.
    int32_t numel() const noexcept {
        return numel_;
    }

    // Resize the tensor to have a new shape. The new shape must not be larger than the
    // shape provided when the tensor was created because this function does not
    // reallocate tensor storage.
    // Note: this is not a reshape function because a reshape function can only reshape
    // a tensor if the new and the existing shapes have the same number of elements.
    void resize(std::initializer_list<int> shape) noexcept;

    friend std::ostream& operator<<(std::ostream& stream, const Tensor& tensor);
    void print() const noexcept;

    void print_info() const noexcept;

    /// Returns the size of the give dimension.
    int32_t size(int32_t i) const;

    bool shape_is_equal(std::initializer_list<int> shape) const noexcept;

    size_t nbytes() const noexcept {
        return numel_ * itemsize();
    }

private:
    // shared ptr allows us to share the same data across many tensors.
    std::shared_ptr<uint8_t[]> data_;

    // Capacity of the data pointer allocated storage, in bytes.
    size_t storage_capacity_{0};

    // Number of elements in the tensor.
    int32_t numel_{0};

    // Number of tensor dimensions.
    int32_t ndims_{0};
    int32_t shape_[3]{0, 0, 0};
    Dtype dtype_{Dtype::Float32};

    int32_t numel_from_shape() const noexcept;
    void set_shape(std::initializer_list<int> shape);
    void print_single(int32_t item_idx, int32_t col_idx, int32_t n_cols) const noexcept;
};

} // namespace gten
