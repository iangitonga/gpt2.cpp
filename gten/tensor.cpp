#include "tensor.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <iomanip>



namespace gten {

Tensor::Tensor(std::initializer_list<int> shape, Dtype dtype)
{
    set_shape(shape);
    dtype_ = dtype;
    ndims_ = shape.size();
    numel_ = numel_from_shape();
    storage_capacity_ = nbytes();
    data_ = std::shared_ptr<uint8_t[]>(new uint8_t[storage_capacity_]);
}

// An empty deleter allows us to use external data storage that we do not own.
static void empty_deleter(uint8_t* ptr) {  }

Tensor::Tensor(void* data_ptr, std::initializer_list<int> shape, Dtype dtype)
{
    GTEN_ASSERT(data_ptr != nullptr, "Expected a non-null pointer but got a nullptr.");
    uint8_t* real_ptr = static_cast<uint8_t*>(data_ptr);
    // An empty deleter ensures we do not delete the data since we do not own it.
    data_ = std::shared_ptr<uint8_t[]>(real_ptr, empty_deleter);
    set_shape(shape);
    dtype_ = dtype;
    ndims_ = shape.size();
    numel_ = numel_from_shape();
    storage_capacity_ = 0;
}

void Tensor::set_shape(std::initializer_list<int> shape) {
    const int ndims = shape.size();
    GTEN_ASSERT(
        ndims > 1 || ndims < 3,
        "Expected tensor shape to have ndims=(1, 2 or 3) but got ndims=%d instead.",
        ndims);

    for (int i = 0; i < ndims; i++)
    {
        int size_i = *(shape.begin() + i);
        shape_[i] = size_i;
        GTEN_ASSERT(size_i != 0, "The size of dim %d of the given shape is zero.", i);
    }
}

void Tensor::resize(std::initializer_list<int> shape) noexcept
{
    set_shape(shape);
    ndims_ = shape.size();
    numel_ = numel_from_shape();
    GTEN_ASSERT(
        nbytes() <= storage_capacity_,
        "Resize size: %ld, exceeds preallocated size: %ld.",
        nbytes(),
        storage_capacity_);
}

int32_t Tensor::numel_from_shape() const noexcept {
    int32_t numel = 1;
    for (int i = 0; i < ndims_; i++)
       numel *= shape_[i];
    return numel;
}

int32_t Tensor::size(int32_t i) const
{
    GTEN_ASSERT(
        i < ndims_,
        "Tensor dim access, %d, is out of range of a tensor with %d-dims.",
        i,
        ndims_);
    return shape_[i];
}

bool Tensor::shape_is_equal(std::initializer_list<int> shape) const noexcept
{
    if (shape.size() != static_cast<size_t>(ndims_))
        return false;
    for (int i = 0; i < ndims_; i++)
        if (shape_[i] != *(shape.begin() + 1))
            return false;
    return true;
}

void Tensor::print_info() const noexcept {
    auto data_pointer = reinterpret_cast<void*>(data_.get());
    std::cout << "Tensor(\n"
              << "  dtype   : " << dtype_str(dtype_) << "\n"
              << "  ndims   : " << ndims_ << "\n"
              << "  shape   : (" << shape_[0] << ", " << shape_[1] << ", " << shape_[2] << ")\n"
              << "  numel   : " << numel_ << "\n"
              << "  capacity: " << storage_capacity_ << "\n"
              << "  pointer : "   << data_pointer << "\n)\n";
}

void Tensor::print_single(int32_t item_idx, int32_t col_idx, int32_t n_cols) const noexcept
{
    uint32_t max_cols = dtype_ == kInt32 ? 32 : 8;
    if (dtype_ == kFloat16)
        std::cout << std::fixed
                  << std::setprecision(4)
                  << std::setw(7)
                  << fp16_to_fp32(reinterpret_cast<Float16*>(data_.get())[item_idx]);
    else if (dtype_ == kFloat32)
        std::cout << std::fixed
                  << std::setprecision(4)
                  << std::setw(7)
                  << reinterpret_cast<Float32*>(data_.get())[item_idx];
    else
        std::cout << reinterpret_cast<Int32*>(data_.get())[item_idx];
    if (col_idx != n_cols - 1)
        std::cout << ", ";
    if (col_idx > 0 && (col_idx % max_cols) == 0)
        std::cout << "\n  ";
}

void Tensor::print() const noexcept
{
    std::cout << "Tensor(\n";

    if (dtype_ == kFloat16)
        std::cout << "Numel=" << numel_ << "\nDtype=Float16\n[";
    else if (dtype_ == kFloat32)
        std::cout << "Numel=" << numel_ << "\nDtype=Float32\n[";
    else
        std::cout << "Numel=" << numel_ << "\nDtype=Int32\n[";

    if (ndims_ == 1)
    {
        for (int col = 0; col < numel_; col += 1)
            print_single(col, col, numel_);
    }
    else if (ndims_ == 2)
    {
        const int n_rows = shape_[0];
        const int n_cols = shape_[1];
        for (int row = 0; row < n_rows; row++)
        {
            if (row == 0) std::cout << "[";
            else std::cout << " [";
            for (int col = 0; col < n_cols; col++)
            {
                const int idx = row * shape_[1] + col;
                if (idx >= numel_)
                    break;
                print_single(idx, col, n_cols);
            }
            if (row != n_rows - 1) std::cout << "]\n";
            else std::cout << "]";
        }
    }
    else // ndims=3
    {
        const int n_depth = shape_[0];
        const int n_rows = shape_[1];
        const int n_cols = shape_[2];
        for (int depth = 0; depth < n_depth; depth++)
        {
            if (depth == 0) std::cout << "[";
            else std::cout << " [";
            for (int row = 0; row < n_rows; row++)
            {
                if (row == 0) std::cout << "[";
                else std::cout << "  [";
                for (int col = 0; col < n_cols; col++)
                {
                    const int idx = (depth * shape_[1] * shape_[2]) + (row * shape_[1]) + col;
                    if (idx >= numel_)
                        break;
                    print_single(idx, col, n_cols);
                }
                std::cout << "]";
                if (row != n_rows - 1)
                    std::cout << "\n";
            }
            std::cout << "]";
            if (depth != n_depth - 1)
                std::cout << "\n\n";
        }
        
    }
    std::cout << "])\n";
}

std::ostream& operator<<(std::ostream& stream, const Tensor& tensor) {
    tensor.print();
    return stream;
}

} // namespace gten
