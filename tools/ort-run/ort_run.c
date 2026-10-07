/* ort_run: run an ONNX/ORT model through the ONNX Runtime C API on raw input files and write every output
 * as float32. Built against the 1.30.0 headers and linked to whichever libonnxruntime.so.1 is on the library
 * path, so the same binary proves each size-study flavour of the library.
 *
 *   ort_run <model.ort|model.onnx> -- <out_prefix> <name>=<file>:<dtype>:<d0,d1,...> ... [-- <out_prefix> ...]
 *   ort_run <model> <input.f32> <d0,d1,...> [output.f32]      (old single-float-input form)
 *
 * dtype: f32 i32 i64 u8 i8 bool. Output i goes to <out_prefix>.<i>.f32. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "onnxruntime_c_api.h"

static const OrtApi* g;
#define CK(x) do { OrtStatus* s_ = (x); if (s_) { fprintf(stderr, "ORT error: %s\n", g->GetErrorMessage(s_)); return 1; } } while (0)

typedef struct { char* name; char* file; ONNXTensorElementDataType t; size_t esz; int64_t shape[8]; size_t nd; } Spec;

static int dtype(const char* d, ONNXTensorElementDataType* t, size_t* esz) {
    if (!strcmp(d, "f32")) { *t = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; *esz = 4; }
    else if (!strcmp(d, "i32")) { *t = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32; *esz = 4; }
    else if (!strcmp(d, "i64")) { *t = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64; *esz = 8; }
    else if (!strcmp(d, "u8")) { *t = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8; *esz = 1; }
    else if (!strcmp(d, "i8")) { *t = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8; *esz = 1; }
    else if (!strcmp(d, "bool")) { *t = ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL; *esz = 1; }
    else return 0;
    return 1;
}

static int parse(char* s, Spec* o) {
    char* eq = strchr(s, '='); char* c2 = strrchr(s, ':'); if (!eq || !c2 || c2 < eq) return 0;
    *c2 = 0; char* c1 = strrchr(s, ':'); if (!c1 || c1 < eq) return 0;
    *eq = 0; *c1 = 0; o->name = s; o->file = eq + 1;
    if (!dtype(c1 + 1, &o->t, &o->esz)) return 0;
    o->nd = 0; for (char* t = strtok(c2 + 1, ","); t && o->nd < 8; t = strtok(NULL, ",")) o->shape[o->nd++] = atoll(t);
    return 1;
}

static float* to_float(const void* p, ONNXTensorElementDataType t, size_t n) {
    float* f = malloc(n * sizeof(float) + 1);
    for (size_t i = 0; i < n; ++i) switch (t) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: f[i] = ((const float*)p)[i]; break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: f[i] = (float)((const int32_t*)p)[i]; break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: f[i] = (float)((const int64_t*)p)[i]; break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: f[i] = ((const uint8_t*)p)[i]; break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: f[i] = ((const int8_t*)p)[i]; break;
        default: f[i] = 0; }
    return f;
}

static void* slurp(const char* path, size_t* nb) {
    FILE* f = fopen(path, "rb"); if (!f) { perror(path); return NULL; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    void* x = malloc(n + 1); if (fread(x, 1, n, f) != (size_t)n) { fclose(f); return NULL; } fclose(f);
    *nb = n; return x;
}

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: ort_run <model> -- <out_prefix> <name>=<file>:<dtype>:<d0,...> ... [-- ...]\n"); return 2; }
    g = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    printf("onnxruntime %s\n", OrtGetApiBase()->GetVersionString());
    OrtEnv* env; CK(g->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "ort_run", &env));
    OrtSessionOptions* so; CK(g->CreateSessionOptions(&so)); CK(g->SetIntraOpNumThreads(so, 2));
    OrtSession* sess; CK(g->CreateSession(env, argv[1], so, &sess));
    OrtAllocator* al; CK(g->GetAllocatorWithDefaultOptions(&al));
    OrtMemoryInfo* mi; CK(g->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mi));
    size_t nin, nout; CK(g->SessionGetInputCount(sess, &nin)); CK(g->SessionGetOutputCount(sess, &nout));
    char** onames = malloc(nout * sizeof(char*));
    for (size_t i = 0; i < nout; ++i) CK(g->SessionGetOutputName(sess, i, al, &onames[i]));
    int legacy = strcmp(argv[2], "--") != 0;
    int a = legacy ? 2 : 3;
    while (a < argc) {
        const char* prefix; Spec specs[16]; size_t ns = 0;
        if (legacy) {
            char *in0; CK(g->SessionGetInputName(sess, 0, al, &in0));
            specs[0].name = in0; specs[0].file = argv[2]; specs[0].t = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; specs[0].esz = 4;
            specs[0].nd = 0; for (char* t = strtok(argv[3], ","); t && specs[0].nd < 8; t = strtok(NULL, ",")) specs[0].shape[specs[0].nd++] = atoll(t);
            ns = 1; prefix = NULL; a = argc;
        } else {
            prefix = argv[a++];
            while (a < argc && strcmp(argv[a], "--") != 0 && ns < 16) {
                if (!parse(argv[a], &specs[ns])) { fprintf(stderr, "bad input spec: %s\n", argv[a]); return 2; }
                ns++; a++;
            }
            if (a < argc) a++;
        }
        OrtValue* ins[16]; const char* in_names[16];
        for (size_t k = 0; k < ns; ++k) {
            size_t nb; void* x = slurp(specs[k].file, &nb); if (!x) return 1;
            CK(g->CreateTensorWithDataAsOrtValue(mi, x, nb, specs[k].shape, specs[k].nd, specs[k].t, &ins[k]));
            in_names[k] = specs[k].name;
        }
        OrtValue** outs = calloc(nout, sizeof(OrtValue*));
        CK(g->Run(sess, NULL, in_names, (const OrtValue* const*)ins, ns, (const char* const*)onames, nout, outs));
        for (size_t i = 0; i < nout; ++i) {
            OrtTensorTypeAndShapeInfo* ti; CK(g->GetTensorTypeAndShape(outs[i], &ti));
            size_t n; CK(g->GetTensorShapeElementCount(ti, &n));
            ONNXTensorElementDataType t; CK(g->GetTensorElementType(ti, &t));
            void* p; CK(g->GetTensorMutableData(outs[i], &p));
            float* y = to_float(p, t, n);
            size_t best = 0; for (size_t j = 1; j < n; ++j) if (y[j] > y[best]) best = j;
            printf("%s out[%zu] %s type=%d n=%zu argmax=%zu max=%.6g\n", prefix ? prefix : "-", i, onames[i], (int)t, n, best, n ? y[best] : 0.0);
            if (legacy && argc > 4 && i == 0) { FILE* o = fopen(argv[4], "wb"); fwrite(y, sizeof(float), n, o); fclose(o); }
            if (prefix) {
                char path[4096]; snprintf(path, sizeof path, "%s.%zu.f32", prefix, i);
                FILE* o = fopen(path, "wb"); fwrite(y, sizeof(float), n, o); fclose(o);
            }
            free(y);
        }
        (void)nin;
    }
    return 0;
}
