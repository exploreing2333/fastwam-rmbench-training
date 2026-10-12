#include <stdio.h>
#include <stdlib.h>
#include <zstd.h>

int main(int argc, char **argv) {
    FILE *input = argc == 2 ? fopen(argv[1], "rb") : stdin;
    if (argc > 2 || input == NULL) { perror("input"); return 2; }
    size_t capacity = 8 * 1024 * 1024;
    void *in_data = malloc(capacity), *out_data = malloc(capacity);
    ZSTD_DStream *stream = ZSTD_createDStream();
    if (!in_data || !out_data || !stream) return 3;
    size_t status = ZSTD_initDStream(stream);
    if (ZSTD_isError(status)) return 4;
    int seen = 0;
    for (;;) {
        size_t bytes = fread(in_data, 1, capacity, input);
        if (bytes == 0) {
            if (ferror(input)) return 5;
            break;
        }
        seen = 1;
        ZSTD_inBuffer in_buffer = {in_data, bytes, 0};
        int pending = 1;
        while (pending) {
            ZSTD_outBuffer out_buffer = {out_data, capacity, 0};
            status = ZSTD_decompressStream(stream, &out_buffer, &in_buffer);
            if (ZSTD_isError(status)) {
                fprintf(stderr, "%s\n", ZSTD_getErrorName(status)); return 6;
            }
            if (fwrite(out_data, 1, out_buffer.pos, stdout) != out_buffer.pos) return 7;
            pending = in_buffer.pos < in_buffer.size || (out_buffer.pos == out_buffer.size && status != 0);
        }
    }
    if (!seen || status != 0) { fprintf(stderr, "Truncated Zstandard archive\n"); return 8; }
    if (fflush(stdout) != 0) return 9;
    ZSTD_freeDStream(stream); free(in_data); free(out_data);
    if (input != stdin) fclose(input);
    return 0;
}
