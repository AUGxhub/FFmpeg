/*
 * QuickTime Graphics (SMC) Video Encoder
 * Copyright (c) 2021 The FFmpeg project
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

/**
 * @file smcenc.c
 * QT SMC Video Encoder by Paul B. Mahol
 */

#include "libavutil/avassert.h"
#include "libavutil/common.h"
#include "libavutil/opt.h"

#include "avcodec.h"
#include "encode.h"
#include "internal.h"
#include "bytestream.h"

#define CPAIR 2
#define CQUAD 4
#define COCTET 8

#define COLORS_PER_TABLE 256

typedef struct SMCContext {
    AVClass *avclass;

    AVFrame *prev_frame;    // buffer for previous source frame
    PutByteContext pb;

    uint8_t mono_value;
    uint8_t distinct_values[16];
    uint8_t color_pairs[COLORS_PER_TABLE * CPAIR];
    uint8_t color_quads[COLORS_PER_TABLE * CQUAD];
    uint8_t color_octets[COLORS_PER_TABLE * COCTET];

    int first_frame;        // flag set to one when the first frame is being processed
                            // so that comparisons with previous frame data in not attempted
    int key_frame;
} SMCContext;

#define ADVANCE_BLOCK(pixel_ptr, row_ptr, nb_blocks) \
{ \
    for (int block = 0; block < nb_blocks && pixel_ptr && row_ptr; block++) { \
        pixel_ptr += 4; \
        if (pixel_ptr - row_ptr >= width) \
        { \
            row_ptr += stride * 4; \
            pixel_ptr = row_ptr; \
        } \
    } \
}

static int smc_cmp_values(const void *a, const void *b)
{
    const uint8_t *aa = a, *bb = b;

    return FFDIFFSIGN(aa[0], bb[0]);
}

static int count_distinct_items(const uint8_t *block_values,
                                uint8_t *distinct_values,
                                int size)
{
    int n = 1;

    distinct_values[0] = block_values[0];
    for (int i = 1; i < size; i++) {
        if (block_values[i] != block_values[i-1]) {
            distinct_values[n] = block_values[i];
            n++;
        }
    }

    return n;
}

static void smc_encode_stream(SMCContext *s, const AVFrame *frame)
{
    PutByteContext *pb = &s->pb;
    const uint8_t *src_pixels = (const uint8_t *)frame->data[0];
    const int stride = frame->linesize[0];
    const uint8_t *prev_pixels = (const uint8_t *)s->prev_frame->data[0];
    const uint8_t *pixel_ptr, *row_ptr;
    const int width = frame->width;
    const uint8_t *old_pixel_ptr = NULL;
    const uint8_t *old_row_ptr = NULL;
    uint8_t block_values[16];
    int block_counter = 0;
    int color_pair_index = 0;
    int color_quad_index = 0;
    int color_octet_index = 0;
    int color_table_index;  /* indexes to color pair, quad, or octet tables */
    int total_blocks;

    s->key_frame = 1;

    /* Number of 4x4 blocks in frame. */
    total_blocks = ((frame->width + 3) / 4) * ((frame->height + 3) / 4);

    pixel_ptr = row_ptr = src_pixels;

    while (block_counter < total_blocks) {
        const uint8_t *xpixel_ptr = pixel_ptr;
        const uint8_t *xrow_ptr = row_ptr;
        int intra_skip_blocks = 0;
        int inter_skip_blocks = 0;
        int distinct = 0;
        int blocks = 0;

        while (s->first_frame == 0 && block_counter + inter_skip_blocks < total_blocks) {
            int compare = 0;

            for (int y = 0; y < 4; y++) {
                const ptrdiff_t offset = pixel_ptr - src_pixels;
                const uint8_t *prev_pixel_ptr = prev_pixels + offset;

                compare |= memcmp(prev_pixel_ptr + y * stride, pixel_ptr + y * stride, 4);
                if (compare)
                    break;
            }

            if (compare)
                break;

            if (inter_skip_blocks >= 256)
                break;
            inter_skip_blocks++;

            s->key_frame = 0;

            ADVANCE_BLOCK(pixel_ptr, row_ptr, 1)
        }

        while (block_counter > 0 && block_counter + intra_skip_blocks < total_blocks &&
               old_pixel_ptr != NULL) {
            int compare = 0;

            for (int y = 0; y < 4; y++) {
                compare |= memcmp(old_pixel_ptr + y * stride, pixel_ptr + y * stride, 4);
                if (compare)
                    break;
            }

            if (compare)
                break;

            if (intra_skip_blocks >= 256)
                break;
            intra_skip_blocks++;
            ADVANCE_BLOCK(pixel_ptr, row_ptr, 1)
        }

        if (intra_skip_blocks == 0 && inter_skip_blocks == 0) {
            if (block_counter + blocks < total_blocks) {
                int nb_distinct;

                for (int y = 0; y < 4; y++)
                    memcpy(block_values + y * 4, pixel_ptr + y * stride, 4);

                qsort(block_values, 16, sizeof(block_values[0]), smc_cmp_values);
                nb_distinct = count_distinct_items(block_values, s->distinct_values, 16);
                s->mono_value = block_values[0];

                distinct = nb_distinct;
                ADVANCE_BLOCK(pixel_ptr, row_ptr, 1)
                blocks++;
            }
        }

        pixel_ptr = xpixel_ptr;
        row_ptr = xrow_ptr;

        if (intra_skip_blocks > 0 && intra_skip_blocks >= inter_skip_blocks) {
            distinct = 17;
            blocks = intra_skip_blocks;
        }

        if (intra_skip_blocks > 16 && intra_skip_blocks >= inter_skip_blocks) {
            distinct = 18;
            blocks = intra_skip_blocks;
        }

        if (inter_skip_blocks > 0 && inter_skip_blocks > intra_skip_blocks) {
            distinct = 19;
            blocks = inter_skip_blocks;
        }

        if (inter_skip_blocks > 16 && inter_skip_blocks > intra_skip_blocks) {
            distinct = 20;
            blocks = inter_skip_blocks;
        }

        if (intra_skip_blocks > 0) {
            old_row_ptr = row_ptr;
            old_pixel_ptr = pixel_ptr;
        }

        switch (distinct) {
        case 1:
            bytestream2_put_byte(pb, 0x60 | (blocks - 1));
            bytestream2_put_byte(pb, s->mono_value);
            ADVANCE_BLOCK(pixel_ptr, row_ptr, blocks)
            break;
        case 2:
            bytestream2_put_byte(pb, 0x80 | (blocks - 1));
            for (int i = 0; i < CPAIR; i++) {
                bytestream2_put_byte(pb, s->distinct_values[i]);

                color_table_index = CPAIR * color_pair_index + i;
                s->color_pairs[s->distinct_values[i]] = color_table_index & 1;
            }

            color_table_index = CPAIR * color_pair_index;
            color_pair_index++;
            if (color_pair_index == COLORS_PER_TABLE)
                color_pair_index = 0;

            for (int i = 0; i < blocks; i++) {
                uint16_t flags = 0;
                int shift = 15;

                for (int y = 0; y < 4; y++) {
                    for (int x = 0; x < 4; x++) {
                        flags |= s->color_pairs[pixel_ptr[x + y * stride]] << shift;
                        shift--;
                    }
                }

                bytestream2_put_be16(pb, flags);

                ADVANCE_BLOCK(pixel_ptr, row_ptr, 1)
            }
            break;
        case 3:
        case 4:
            bytestream2_put_byte(pb, 0xA0 | (blocks - 1));
            for (int i = 0; i < CQUAD; i++) {
                bytestream2_put_byte(pb, s->distinct_values[i]);

                color_table_index = CQUAD * color_quad_index + i;
                s->color_quads[s->distinct_values[i]] = color_table_index & 3;
            }

            color_table_index = CQUAD * color_quad_index;
            color_quad_index++;
            if (color_quad_index == COLORS_PER_TABLE)
                color_quad_index = 0;

            for (int i = 0; i < blocks; i++) {
                uint32_t flags = 0;
                int shift = 30;

                for (int y = 0; y < 4; y++) {
                    for (int x = 0; x < 4; x++) {
                        flags |= s->color_quads[pixel_ptr[x + y * stride]] << shift;
                        shift -= 2;
                    }
                }

                bytestream2_put_be32(pb, flags);

                ADVANCE_BLOCK(pixel_ptr, row_ptr, 1)
            }
            break;
        case 25:
        case 26:
        case 27:
        case 28:
            bytestream2_put_byte(pb, 0xD0 | (blocks - 1));
            bytestream2_put_byte(pb, block_values[0]);
            for (int i = 0; i < blocks; i++) {
                bytestream2_put_be16(pb, 0);
                bytestream2_put_be16(pb, 0);
                bytestream2_put_be16(pb, 0);
            }
            ADVANCE_BLOCK(pixel_ptr, row_ptr, blocks)
            break;
        default:
            bytestream2_put_byte(pb, 0xE0 | (blocks - 1));
            for (int i = 0; i < blocks; i++) {
                for (int y = 0; y < 4; y++) {
                    for (int x = 0; x < 4; x++)
                        bytestream2_put_byte(pb, pixel_ptr[x + y * stride]);
                }

                ADVANCE_BLOCK(pixel_ptr, row_ptr, 1)
            }
            break;
        case 17:
            bytestream2_put_byte(pb, 0x20 | (blocks - 1));
            ADVANCE_BLOCK(pixel_ptr, row_ptr, blocks)
            break;
        case 18:
            bytestream2_put_byte(pb, 0x30);
            bytestream2_put_byte(pb, blocks - 1);
            ADVANCE_BLOCK(pixel_ptr, row_ptr, blocks)
            break;
        case 19:
            bytestream2_put_byte(pb, 0x00 | (blocks - 1));
            ADVANCE_BLOCK(pixel_ptr, row_ptr, blocks)
            break;
        case 20:
            bytestream2_put_byte(pb, 0x10);
            bytestream2_put_byte(pb, blocks - 1);
            ADVANCE_BLOCK(pixel_ptr, row_ptr, blocks)
            break;
        }

        block_counter += blocks;
    }
}

static int smc_encode_init(AVCodecContext *avctx)
{
    SMCContext *s = avctx->priv_data;

    avctx->bits_per_coded_sample = 8;

    s->prev_frame = av_frame_alloc();
    if (!s->prev_frame)
        return AVERROR(ENOMEM);

    return 0;
}

static int smc_encode_frame(AVCodecContext *avctx, AVPacket *pkt,
                                const AVFrame *frame, int *got_packet)
{
    SMCContext *s = avctx->priv_data;
    const AVFrame *pict = frame;
    uint8_t *buf;
    int ret = ff_alloc_packet(avctx, pkt, 8LL * avctx->height * avctx->width);
    uint8_t *pal;

    if (ret < 0)
        return ret;

    bytestream2_init_writer(&s->pb, pkt->data, pkt->size);

    // skip 4 byte header, write it later once the size of the chunk is known
    bytestream2_put_be32(&s->pb, 0x00);

    if (!s->prev_frame->data[0]) {
        s->first_frame = 1;
        s->prev_frame->format = pict->format;
        s->prev_frame->width = pict->width;
        s->prev_frame->height = pict->height;
        ret = av_frame_get_buffer(s->prev_frame, 0);
        if (ret < 0)
            return ret;
    } else {
        s->first_frame = 0;
    }

    pal = av_packet_new_side_data(pkt, AV_PKT_DATA_PALETTE, AVPALETTE_SIZE);
    memcpy(pal, frame->data[1], AVPALETTE_SIZE);

    smc_encode_stream(s, pict);

    av_shrink_packet(pkt, bytestream2_tell_p(&s->pb));
    buf = pkt->data;

    buf[0] = 0x0;

    // write chunk length
    AV_WB24(buf + 1, pkt->size);

    av_frame_unref(s->prev_frame);
    ret = av_frame_ref(s->prev_frame, frame);
    if (ret < 0) {
        av_log(avctx, AV_LOG_ERROR, "cannot add reference\n");
        return ret;
    }

    if (s->key_frame)
        pkt->flags |= AV_PKT_FLAG_KEY;

    *got_packet = 1;

    return 0;
}

static int smc_encode_end(AVCodecContext *avctx)
{
    SMCContext *s = (SMCContext *)avctx->priv_data;

    av_frame_free(&s->prev_frame);

    return 0;
}

#define OFFSET(x) offsetof(SMCContext, x)
#define VE AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_ENCODING_PARAM
static const AVOption options[] = {
    { NULL },
};

static const AVClass smc_class = {
    .class_name = "smc",
    .item_name  = av_default_item_name,
    .option     = options,
    .version    = LIBAVUTIL_VERSION_INT,
};

const AVCodec ff_smc_encoder = {
    .name           = "smc",
    .long_name      = NULL_IF_CONFIG_SMALL("QuickTime Graphics (SMC)"),
    .type           = AVMEDIA_TYPE_VIDEO,
    .id             = AV_CODEC_ID_SMC,
    .priv_data_size = sizeof(SMCContext),
    .priv_class     = &smc_class,
    .init           = smc_encode_init,
    .encode2        = smc_encode_frame,
    .close          = smc_encode_end,
    .caps_internal  = FF_CODEC_CAP_INIT_THREADSAFE,
    .pix_fmts       = (const enum AVPixelFormat[]) { AV_PIX_FMT_PAL8,
                                                     AV_PIX_FMT_NONE},
};
