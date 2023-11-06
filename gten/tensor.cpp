#include <iostream>
#include <iomanip>

#include "tensor.h"


namespace gten {

/*

Types of tensors:
- Weight tensor:
  ~ Allocate -> fill. Always static. 
- Activation tensor:
  ~ Allocate max size required for inference.
  ~ initial numel=0, shape=()
  ~ compute; update [numel, shape, strides]
  ~ compute with offsets.
*/


Tensor::Tensor(const std::vector<int>& shape, TensorDtype dtype, float qscale, int qzerop)
    : dtype_{dtype}, qscale_{qscale}, qzerop_{qzerop}
{
    validate_shape(shape);
    // if (dtype == kQint8) {
    //     GTEN_ASSERT(qscale != 0, "Expected non-zero scale for dtype Qint8 but got %d.", qscale);
    // }
    const int numel = numel_from_shape(shape);
    const int alloc_bytes = numel * itemsize();

    data_ptr_ = std::shared_ptr<uint8_t[]>(new uint8_t[alloc_bytes]);
    storage_size_ = alloc_bytes;
    G_TensorMemAllocated += alloc_bytes;
    numel_ = numel;
    shape_ = shape;
    set_strides_from_shape(shape);
}


// An empty deleter allows us to use external data storage that we do not own.
static void empty_deleter(uint8_t* ptr) {  }

Tensor::Tensor(void* data_ptr, const std::vector<int>& shape, TensorDtype dtype, float qscale, int qzerop)
    : dtype_{dtype}, qscale_{qscale}, qzerop_{qzerop}
{
    // if (dtype == kQint8) {
    //     GTEN_ASSERT(qscale != 0, "Expected non-zero scale for dtype Qint8 but got %d.", qscale);
    // }
    GTEN_ASSERT(data_ptr != nullptr, "Expected a non-null pointer but got a nullptr.");
    uint8_t* real_ptr = static_cast<uint8_t*>(data_ptr);
    // An empty deleter ensures we do not delete the data since we do not own it.
    data_ptr_ = std::shared_ptr<uint8_t[]>(real_ptr, empty_deleter);
    validate_shape(shape);
    shape_ = shape;
    set_strides_from_shape(shape);
    numel_ = numel_from_shape(shape);
    storage_size_ = 0;
}


void Tensor::validate_shape(const std::vector<int>& shape) const {
    GTEN_ASSERT(shape.size() != 0, "The given shape is empty.");
    GTEN_ASSERT(shape.size() <= 3, "Shape with dimensions > 3 not supported.");
    for (int i = 0; i < int(shape.size()); i++) {
        if (shape[i] <= 0) {
            std::cerr << "err\n";
            GTEN_ASSERT(false, "The value of dimension %d: %d of the given shape is invalid!", i, shape[i]);
        }
    }
}


int Tensor::numel_from_shape(const std::vector<int>& shape) const {
    int numel = 1;
    for (int size : shape) {
        numel = numel * size;
    }
    return numel;
}

// Contigous only???
void Tensor::resize(const std::vector<int>& new_shape) {
    validate_shape(new_shape);
    const int new_size = numel_from_shape(new_shape) * itemsize();
    if (new_size > storage_size_) {
        GTEN_ASSERT(false, "The new shape provided (cap=%d) exceeds storage capacity = %d.", new_size, storage_size_);
    }
    shape_ = new_shape;
    set_strides_from_shape(new_shape);
    numel_ = numel_from_shape(new_shape);
}


void Tensor::set_strides_from_shape(const std::vector<int>& shape) {
    // 1-dim: 1
    // 2-dim: d2, 1
    // 3-dim: d2*d3, d3, 1
    switch (shape.size()) {
        case 1: {
            strides_ = {1};
            break;
        }
        case 2: {
            const int d1 = shape[1];
            strides_ = {d1, 1};
            break;
        }
        case 3: {
            const int d1 = shape[1];
            const int d2 = shape[2];
            strides_ = {d1*d2, d2, 1};
            break;
        }
    }
}


void print_vector(const std::vector<int>& vec) {
    std::cout << "(";
    for (int i = 0; i < int(vec.size()); i++) {
        std::cout << vec[i];
        if (i != int(vec.size()) - 1) {
            std::cout << ", ";
        }
    }
    std::cout << ")\n";
}

void Tensor::print_info() const {
    auto data = data_ptr<void>();
    std::cout << "\nTensor(\n"
              << "  dtype    : " << dtype_str(dtype_) << "\n"
              << "  shape    : ";
    print_vector(shape_);
    std::cout << "  strides  : ";
    print_vector(strides_);
    std::cout << "  numel    : " << numel_ << "\n"
            //   << "  numel cap: " << storage_size_/itemsize() << "\n"
              << "  capacity : " << storage_size_ << " bytes\n"
              << "  pointer  : "   << data << "\n)\n";
    
}

// Should we create and return a new tensor with the new shape?
Tensor Tensor::view(const std::vector<int>& new_shape) const {
    validate_shape(new_shape);
    const int new_numel = numel_from_shape(new_shape);
    const int old_numel = numel_from_shape(shape_);
    GTEN_ASSERT(new_numel == old_numel, "New shape numel `%d` must be equal with old shape numel `%d`.", new_numel, old_numel);

    Tensor out = *this;
    out.shape_ = new_shape;
    out.set_strides_from_shape(new_shape);

    return out;
}


// Should we create and return a new tensor with the new shape?
Tensor Tensor::permute(const std::vector<int> &indices)
{
    GTEN_ASSERT(indices.size() == shape_.size(),
                "The dims of indices `%ld` given do not match the tensor dims `%ld`.",
                indices.size(), shape_.size());

    std::vector<int> new_shape = shape_;
    std::vector<int> new_strides = strides_;
    for (int i = 0; i < int(indices.size()); i++) {
        const int idx = indices[i];
        new_shape[i] = shape_[idx];
        new_strides[i] = strides_[idx]; 
    }
    shape_ = std::move(new_shape);
    strides_ = std::move(new_strides);
    
    return *this;
}

void Tensor::set_strides(const std::vector<int>& strides)
{
    GTEN_ASSERT(strides.size() == shape_.size(), "The given strides ndims must match shape ndims.");
    for (int i = 0; i < int(strides.size()); i++) {
        if (strides[i] <= 0) {
            GTEN_ASSERT(false, "The stride at index %d, `%d` is invalid.", i, strides[i]);
        }
    }
    strides_ = strides;
}

std::string Tensor::shape_str() const
{
    std::stringstream s;
    s << "(";
    for (int i = 0; i < int(shape_.size()); i++) {
        s << shape_[i];
        if (i != int(shape_.size()) - 1) {
            s << ", ";
        }
    }
    s << ")";
    
    return s.str();
}

std::string Tensor::strides_str() const {
    std::stringstream s;
    s << "(";
    for (int i = 0; i < int(strides_.size()); i++) {
        s << strides_[i];
        if (i != int(strides_.size()) - 1) {
            s << ", ";
        }
    }
    s << ")";
    
    return s.str();
}

void Tensor::print_single(int item_idx, int col_idx, int n_cols) const
{
    uint32_t max_cols = dtype_ == kInt32 ? 32 : 8;
    if (dtype_ == kFloat16) {
        std::cout << std::fixed
                  << std::setprecision(4)
                  << std::setw(7)
                  << fp16_to_fp32(data_ptr<Float16>()[item_idx]);
    }
    else if (dtype_ == kFloat32) {
        std::cout << std::fixed
                  << std::setprecision(4)
                  << std::setw(7)
                  << data_ptr<float>()[item_idx];
    }
    else {
        std::cout << std::setw(2) << data_ptr<int>()[item_idx];
    }
    if (col_idx != n_cols - 1) {
        std::cout << ", ";
    }
    if (col_idx > 0 && (col_idx % max_cols) == 0) {
        std::cout << "\n  ";
    }
}

void Tensor::print() const
{
    std::cout << "\n[";
    const int ndims = shape_.size();
    if (ndims == 1) {
        for (int col = 0; col < numel_; col += 1)
            print_single(col, col, numel_);
    }
    else if (ndims == 2) {
        const int rows = shape_[0];
        const int cols = shape_[1];
        const int st0 = strides_[0];
        const int st1 = strides_[1];
        for (int row = 0; row < rows; row++) {
            if (row == 0) std::cout << "[";
            else std::cout << " [";
            for (int col = 0; col < cols; col++) {
                const int idx = row * st0 + col * st1;
                print_single(idx, col, cols);
            }
            if (row != rows - 1) std::cout << "]\n";
            else std::cout << "]";
        }
    }
    else // ndims=3
    {
        const int chs = shape_[0];
        const int rows = shape_[1];
        const int cols = shape_[2];
        const int st0 = strides_[0];
        const int st1 = strides_[1];
        const int st2 = strides_[2];

        for (int ch = 0; ch < chs; ch++)
        {
            if (ch == 0) std::cout << "[";
            else std::cout << " [";
            for (int row = 0; row < rows; row++) {
                if (row == 0) std::cout << "[";
                else std::cout << "  [";
                for (int col = 0; col < cols; col++) {
                    const int idx = ch * st0 + row * st1 + col * st2;
                    print_single(idx, col, cols);
                }
                std::cout << "]";
                if (row != rows - 1)
                    std::cout << "\n";
            }
            std::cout << "]";
            if (ch != chs - 1)
                std::cout << "\n\n";
        }
        
    }

    std::cout << "]\n\n";
}

std::ostream& operator<<(std::ostream& stream, const Tensor& tensor) {
    tensor.print();
    return stream;
}

} // namespace gten.
