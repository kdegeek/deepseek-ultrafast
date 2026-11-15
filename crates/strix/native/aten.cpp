// Optional native compatibility kernels. No Python headers, symbols or runtime.
// Equations follow pinned model.py and ports/strix-reference/kernel.py verbatim.
#include <ATen/ATen.h>
#include <c10/core/InferenceMode.h>
#include <c10/core/CachingDeviceAllocator.h>
#include <torch/version.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#define API extern "C" __declspec(dllexport)
using Tensor = at::Tensor;
struct Handle { Tensor value; };

template<class F> int guarded(char* error, size_t capacity, F&& fn) noexcept {
    if (!error || capacity == 0) return 1;
    error[0] = 0;
    try { c10::InferenceMode inference; fn(); return 0; }
    catch (const std::exception& e) { std::snprintf(error, capacity, "%s", e.what()); }
    catch (...) { std::snprintf(error, capacity, "unknown native ATen exception"); }
    return 1;
}
void require(bool valid, const char* text) { if (!valid) throw std::runtime_error(text); }
Tensor gpu(const float* data, int64_t count, at::IntArrayRef shape) {
    require(data && count > 0, "empty input");
    return at::from_blob(const_cast<float*>(data), {count}, at::kFloat)
        .reshape(shape).to(at::Device(at::kCUDA, 0)); // synchronous, owns its GPU storage
}
API unsigned strix_aten_abi() noexcept { return 1; }
API const char* strix_aten_version() noexcept { return TORCH_VERSION; }
API int strix_aten_memory(int64_t* output, char* error, size_t capacity) noexcept {
    return guarded(error,capacity,[&] {
        require(output,"memory output pointer");
        auto stats=c10::getDeviceAllocator(at::kCUDA)->getDeviceStats(0);
        output[0]=stats.allocated_bytes[0].current; output[1]=stats.allocated_bytes[0].peak;
        output[2]=stats.reserved_bytes[0].current; output[3]=stats.reserved_bytes[0].peak;
    });
}
API int strix_aten_create(const float* data, int64_t count, const int64_t* shape,
                         int64_t rank, Handle** output, char* error, size_t capacity) noexcept {
    if (output) *output = nullptr;
    return guarded(error, capacity, [&] {
        require(data && output && shape && rank > 0 && rank <= 4 && count > 0, "invalid tensor arguments");
        int64_t size = 1;
        for (int64_t i=0; i<rank; ++i) {
            require(shape[i] > 0 && size <= INT64_MAX / shape[i], "invalid tensor shape");
            size *= shape[i];
        }
        require(size == count, "tensor shape/count mismatch");
        for (int64_t i=0; i<count; ++i) require(std::isfinite(data[i]), "nonfinite input");
        auto value = gpu(data, count, {shape, static_cast<size_t>(rank)});
        *output = new Handle{std::move(value)};
    });
}
API void strix_aten_destroy(Handle* value) noexcept { delete value; }
API int strix_aten_bytes(const uint8_t* data, int64_t rows, int64_t cols,
                        Handle** output, char* error, size_t capacity) noexcept {
    if(output) *output=nullptr;
    return guarded(error,capacity,[&] {
        require(data && output && rows>0 && cols>0 && rows<=INT64_MAX/cols && rows*cols<=64*1024*1024,"packed upload size");
        auto value=at::from_blob(const_cast<uint8_t*>(data),{rows,cols},at::kByte).to(at::Device(at::kCUDA,0));
        *output=new Handle{std::move(value)};
    });
}
API int strix_aten_copy(const Handle* value, float* output, int64_t count,
                       char* error, size_t capacity) noexcept {
    return guarded(error, capacity, [&] {
        require(value && output && count == value->value.numel(), "copy shape mismatch");
        auto host = value->value.to(at::kFloat).cpu().contiguous();
        require(at::isfinite(host).all().item<bool>(), "nonfinite ATen result");
        std::memcpy(output, host.const_data_ptr<float>(), count * sizeof(float));
    });
}
// op 0 linear; 1 complete HC; 2 RMS; 3 pre; 4 post. All inputs stay on GPU.
API int strix_aten_run(int op, const Handle* const* inputs, int64_t count,
                      const double* params, int64_t nparams, Handle** output,
                      char* error, size_t capacity) noexcept {
    if (output) *output = nullptr;
    return guarded(error, capacity, [&] {
        require(output && inputs && count > 0 && count <= 8 && nparams >= 0 && nparams <= 8,
                "invalid operation arguments");
        std::vector<Tensor> a;
        for (int64_t i=0; i<count; ++i) {
            require(inputs[i] && inputs[i]->value.is_cuda(), "HIP tensors required");
            a.push_back(inputs[i]->value);
        }
        auto p = [&](int64_t i) { require(params && i<nparams && std::isfinite(params[i]), "missing parameter"); return params[i]; };
        // Validate before converting untrusted floating ABI parameters to an
        // integer; an out-of-range C++ floating-to-integer cast is undefined.
        auto integer = [&](int64_t i) {
            const double value=p(i);
            require(value>=0 && value<=1073741824. && std::floor(value)==value,"invalid integer parameter");
            return static_cast<int64_t>(value);
        };
        auto arity = [&](size_t n) { require(a.size()==n, "operation arity mismatch"); };
        Tensor result;
        switch (op) {
        case 0: arity(2); result = at::linear(a[0], a[1]); break;
        case 1: {
            arity(4); auto x=a[0].reshape({1,1,-1});
            require(a[1].dim()==2 && a[1].size(0)==24 && a[1].size(1)==x.size(-1)
                    && a[2].numel()==3 && a[3].numel()==24, "HC shape mismatch");
            const auto eps=p(0), norm_eps=p(1); const auto iters=integer(2);
            require(eps>0 && norm_eps>0 && iters>0 && iters<=100, "HC parameters");
            auto inv=at::rsqrt(x.square().mean(-1,true)+norm_eps);
            auto raw=at::linear(x,a[1]); auto z=raw*inv;
            auto scale=a[2].flatten(), base=a[3].flatten();
            auto pre=at::sigmoid(z.slice(-1,0,4)*scale[0]+base.slice(0,0,4))+eps;
            auto post=2*at::sigmoid(z.slice(-1,4,8)*scale[1]+base.slice(0,4,8));
            auto comb=(z.slice(-1,8)*scale[2]+base.slice(0,8)).reshape({1,1,4,4});
            comb=at::softmax(comb,-1)+eps;
            comb=comb/(comb.sum(-2,true)+eps);
            for(int i=1;i<iters;++i) {
                comb=comb/(comb.sum(-1,true)+eps);
                comb=comb/(comb.sum(-2,true)+eps);
            }
            result=at::cat({raw.flatten(),inv.flatten(),z.flatten(),pre.flatten(),post.flatten(),comb.flatten()});
            break;
        }
        case 2: {
            arity(2); auto x=a[0].reshape({1,1,-1});
            require(a[1].numel()==x.numel() && p(0)>0, "RMS shape/epsilon");
            auto inv=at::rsqrt(x.square().mean(-1,true)+p(0));
            result=at::cat({inv.flatten(),(a[1]*(x*inv)).to(at::kBFloat16).to(at::kFloat).flatten()});
            break;
        }
        case 3: {
            arity(2); require(a[1].numel()==4 && a[0].numel()%4==0,"pre shape");
            result=(a[1].reshape({1,1,4,1})*a[0].reshape({1,1,4,-1})).sum(2).to(at::kBFloat16);
            break;
        }
        case 4: {
            arity(4); auto x=a[0].reshape({1,1,-1});
            require(a[1].numel()==4*x.numel() && a[2].numel()==4 && a[3].numel()==16,"post shape");
            auto residual=a[1].reshape({1,1,4,-1}),post=a[2].reshape({1,1,4}),comb=a[3].reshape({1,1,4,4});
            result=(post.unsqueeze(-1)*x.unsqueeze(-2)+(comb.unsqueeze(-1)*residual.unsqueeze(-2)).sum(2)).to(at::kBFloat16);
            break;
        }
        case 5: {
            arity(2); auto x=a[0].reshape({1,-1}), w=a[1];
            const auto mode=integer(0);
            require(w.dim()==2 && x.size(-1)==w.size(-1) && mode>=0 && mode<=2,"linear shape/mode");
            if(mode==0) result=at::matmul(x,w.t()).to(at::kBFloat16);
            else if(mode==1) result=at::linear(x.to(at::kBFloat16),w.to(at::kBFloat16));
            else result=at::linear(x,w);
            break;
        }
        case 6: {
            arity(4); const int64_t d=integer(0);
            require(d>0 && a[0].numel()==4*d && a[1].numel()==5*d && a[2].numel()==4*d && a[3].numel()==4*d,"Engram shape");
            auto h=a[0].reshape({1,1,4,d}); auto kv=a[1].flatten();
            auto key=kv.slice(0,0,4*d).reshape({1,1,4,d}), value=kv.slice(0,4*d).reshape({1,1,d});
            auto weight=a[2].reshape({4,d})*a[3].reshape({4,d});
            auto rstd=at::rsqrt(h.square().mean(-1)+p(1))*at::rsqrt(key.square().mean(-1)+p(1));
            auto dot=(h*weight*key).sum(-1)*rstd*std::pow(double(d),-0.5);
            auto gate=at::sigmoid(at::copysign(dot.abs().clamp_min(1e-6).sqrt(),dot));
            result=(h+gate.unsqueeze(-1)*value.unsqueeze(-2)).to(at::kBFloat16);
            break;
        }
        case 7: {
            arity(2); require(a[0].numel()==a[1].numel(),"SwiGLU shape");
            auto gate=a[0],up=a[1];
            if(p(0)>0) { gate=gate.clamp_max(p(0)); up=up.clamp(-p(0),p(0)); }
            result=(p(1)*(at::silu(gate)*up)).to(at::kBFloat16);
            break;
        }
        case 8: case 14: {
            arity(2); auto logits=a[0].reshape({1,-1}),bias=a[1].reshape({1,-1});
            require(logits.sizes()==bias.sizes() && p(0)>0 && p(0)<=logits.numel(),"router shape");
            auto scores=at::softplus(logits).sqrt();
            auto corrected=scores+bias;
            auto ids=std::get<1>(at::topk(corrected,integer(0),-1));
            auto weights=scores.gather(1,ids);
            weights=weights/(weights.sum(-1,true)+1e-20); weights=weights*p(1);
            result=at::cat({ids.to(at::kFloat).flatten(),weights.flatten()});
            if(op==14) result=at::cat({scores.flatten(),corrected.flatten(),result});
            break;
        }
        case 9: {
            arity(2); const int64_t d=integer(0);
            require(d>0 && a[0].numel()%d==0 && a[0].numel()==a[1].numel(),"pool shape");
            auto kv=a[0].reshape({1,-1,d}),score=a[1].reshape({1,-1,d});
            result=(kv*at::softmax(score,1)).sum(1,true).to(at::kBFloat16);
            break;
        }
        case 10: {
            arity(4); const int64_t d=integer(0);
            require(d>0 && a[0].numel()==a[2].numel()*d && a[1].numel()==a[3].numel()*d,"attention shape");
            auto query=a[0].reshape({1,-1,d}),kv=a[1].reshape({1,-1,d});
            auto valid=a[3].reshape({1,-1}).to(at::kBool); const auto heads=a[2].numel();
            auto maximum=at::full({1,heads},-1e30,a[0].options()),denominator=at::zeros_like(maximum);
            auto numerator=at::zeros({1,heads,d},a[0].options());
            for(int64_t offset=0;offset<kv.size(1);offset+=64) {
                auto values=kv.slice(1,offset,offset+64), mask=valid.slice(1,offset,offset+64);
                auto score=at::matmul(query,values.transpose(-1,-2))*std::pow(double(d),-0.5);
                score.masked_fill_(mask.logical_not().unsqueeze(1),-std::numeric_limits<double>::infinity());
                auto new_max=at::maximum(maximum,score.amax(-1));
                auto correction=at::exp(maximum-new_max),probability=at::exp(score-new_max.unsqueeze(-1));
                denominator=denominator*correction+probability.sum(-1);
                numerator=numerator*correction.unsqueeze(-1)+at::matmul(probability.to(at::kBFloat16).to(at::kFloat),values);
                maximum=new_max;
            }
            auto sink=a[2].reshape({1,heads}); auto final_max=at::maximum(maximum,sink);
            auto correction=at::exp(maximum-final_max);
            denominator=denominator*correction+at::exp(sink-final_max);
            numerator=numerator*correction.unsqueeze(-1);
            result=(numerator/denominator.clamp_min(std::numeric_limits<float>::min()).unsqueeze(-1)).to(at::kBFloat16);
            break;
        }
        case 11: {
            arity(1); const auto rows=integer(0),cols=integer(1);
            require(rows>0 && cols>0 && rows<=INT64_MAX/cols && rows*cols<=1024LL*1024*1024,"allocation size");
            result=at::empty({rows,cols},a[0].options()); break;
        }
        case 12: {
            arity(2); const auto start=integer(0);
            require(a[0].dim()==2 && a[1].dim()==2 && a[0].size(1)==a[1].size(1)
                    && start>=0 && start+a[1].size(0)<=a[0].size(0),"upload slice shape");
            a[0].slice(0,start,start+a[1].size(0)).copy_(a[1]); result=a[0]; break;
        }
        case 13: {
            arity(3); const bool fp4=p(0)==4;
            require((fp4 || p(0)==8) && a[1].scalar_type()==at::kByte && a[2].scalar_type()==at::kByte,"packed types");
            auto code=a[1].to(at::kLong);
            if(fp4) code=at::stack({at::bitwise_and(code,15),at::bitwise_right_shift(code,4)},-1).flatten(-2);
            require(code.size(1)==a[0].numel() && a[2].size(1)*32==code.size(1)
                && a[2].size(0)==(fp4?code.size(0):(code.size(0)+31)/32),"packed shape");
            std::vector<float> values;
            if(fp4) values={0.,0.5,1.,1.5,2.,3.,4.,6.};
            else { for(int e=0;e<16;++e) for(int m=0;m<8;++m)
                values.push_back(e==0?float(m)/512:float((1.+m/8.)*std::pow(2.,e-7)));
                values.back()=std::numeric_limits<float>::quiet_NaN(); }
            auto table=gpu(values.data(),values.size(),{static_cast<int64_t>(values.size())});
            auto index=at::bitwise_and(code,fp4?7:127);
            auto bv=table.index_select(0,index.flatten()).reshape(code.sizes())
                *at::where(at::bitwise_and(code,fp4?8:128).ne(0),-1.,1.);
            auto exponent=a[2].to(at::kInt)-127;
            auto scale=at::ldexp(at::ones_like(exponent,at::kFloat),exponent);
            if(!fp4) {
                auto ids=at::arange(code.size(0),code.options())/32;
                scale=scale.index_select(0,ids.to(at::kLong));
            }
            bv=(bv.reshape({code.size(0),-1,32})*scale.unsqueeze(-1)).reshape(code.sizes());
            if(p(1)!=0) result=at::linear(a[0].reshape({1,-1}).to(at::kBFloat16),bv.to(at::kBFloat16));
            else result=at::matmul(a[0].reshape({1,-1}),bv.t()).to(at::kBFloat16);
            break;
        }
        case 15: {
            arity(1); const auto dim=integer(0),len=integer(1);
            require(dim>0 && dim%2==0 && len>0 && len<=128,"rotary table shape");
            // The unchanged reference constructs constant buffers on CPU before
            // moving them to HIP. No model weight/input arithmetic runs here.
            auto options=at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
            auto freqs=1.0/at::pow(p(3),at::arange(0,dim,2,options)/dim);
            if(p(2)>0) {
                auto corrected=[&](double rotations) {return dim*std::log(p(2)/(rotations*2*std::acos(-1.)))/(2*std::log(p(3)));};
                const auto low=std::max(std::floor(corrected(p(5))),0.);
                const auto high=std::min(std::ceil(corrected(p(6))),double(dim-1));
                auto ramp=((at::arange(dim/2,options)-low)/std::max(high-low,1e-3)).clamp(0,1);
                auto smooth=1-ramp;
                freqs=freqs/p(4)*(1-smooth)+freqs*smooth;
            }
            auto angles=at::outer(at::arange(len,options.dtype(at::kLong)),freqs);
            result=at::polar(at::ones_like(angles),angles).to(at::Device(at::kCUDA,0));
            break;
        }
        case 16: case 17: {
            arity(op==16?2:3);
            const auto hd=integer(0),rd=integer(1),pos=integer(2);
            require(hd>0 && rd>0 && rd<=hd && rd%2==0 && a[0].numel()%hd==0,"rotary shape");
            Tensor freq;
            if(op==16) {
                require(a[1].is_complex() && a[1].dim()==2 && pos>=0 && pos<a[1].size(0) && a[1].size(1)==rd/2,"rotary table position");
                freq=a[1].select(0,pos);
            } else {
                require(a[1].numel()==rd/2 && a[2].numel()==rd/2,"rotary frequencies");
                freq=at::complex(a[1],a[2]);
            }
            if(p(3)!=0) freq=freq.conj();
            result=a[0].clone().reshape({1,1,-1,hd});
            auto tail=result.slice(-1,hd-rd);
            auto pairs=tail.to(at::kBFloat16).to(at::kFloat).reshape({1,1,-1,rd/2,2});
            auto rotated=at::view_as_real(at::view_as_complex(pairs)*freq.reshape({1,1,1,rd/2})).flatten(-2);
            tail.copy_(rotated.to(at::kBFloat16));
            break;
        }
        default: throw std::runtime_error("unsupported native ATen operation");
        }
        require(result.is_cuda(), "CPU result forbidden");
        *output = new Handle{std::move(result)};
    });
}
