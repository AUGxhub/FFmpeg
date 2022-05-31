/*
 * QOI image format
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

#include <stdlib.h>
#include <string.h>

#include "libavutil/avassert.h"
#include "libavutil/imgutils.h"
#include "libavutil/avstring.h"
#include "avcodec.h"
#include "internal.h"
#include "bytestream.h"
#include "codec_internal.h"
#include "thread.h"

#define QOI_OP_INDEX  0x00 /* 00xxxxxx */
#define QOI_OP_DIFF   0x40 /* 01xxxxxx */
#define QOI_OP_LUMA   0x80 /* 10xxxxxx */
#define QOI_OP_RUN    0xc0 /* 11xxxxxx */
#define QOI_OP_RGB    0xfe /* 11111110 */
#define QOI_OP_RGBA   0xff /* 11111111 */

#define QOI_MASK_2    0xc0 /* 11000000 */

#define QOI_COLOR_HASH(px) (px[0]*3 + px[1]*5 + px[2]*7 + px[3]*11)

static int qoi_decode_frame(AVCodecContext *avctx, AVFrame *p,
                            int *got_frame, AVPacket *avpkt)
{
    const uint8_t *buf = avpkt->data;
    int ret, buf_size = avpkt->size;
    int width, height, run = 0;
    uint8_t index[64][4] = { 0 };
    uint8_t px[4] = { 0, 0, 0, 255 };
    GetByteContext gb;
    uint8_t *dst;
    uint64_t len;

    if (buf_size < 20)
        return AVERROR_INVALIDDATA;

    bytestream2_init(&gb, buf, buf_size);
    bytestream2_skip(&gb, 4);
    width  = bytestream2_get_be32(&gb);
    height = bytestream2_get_be32(&gb);
    bytestream2_skip(&gb, 2);

    if ((ret = ff_set_dimensions(avctx, width, height)) < 0)
        return ret;

    if ((ret = av_image_check_size(avctx->width, avctx->height, 0, NULL)) < 0)
        return ret;

    avctx->pix_fmt = AV_PIX_FMT_RGBA;

    if ((ret = ff_thread_get_buffer(avctx, p, 0)) < 0)
        return ret;

    dst = p->data[0];
    len = width * height * 4LL;
    for (int n = 0, off_x = 0, off_y = 0; n < len; n += 4, off_x++) {
        if (off_x >= width) {
            off_x = 0;
            off_y++;
            dst += p->linesize[0];
        }
        if (run > 0) {
            run--;
        } else if (bytestream2_get_bytes_left(&gb) > 4) {
            int chunk = bytestream2_get_byteu(&gb);

            if (chunk == QOI_OP_RGB) {
                bytestream2_get_bufferu(&gb, px, 3);
            } else if (chunk == QOI_OP_RGBA) {
                bytestream2_get_bufferu(&gb, px, 4);
            } else if ((chunk & QOI_MASK_2) == QOI_OP_INDEX) {
                memcpy(px, index[chunk], 4);
            } else if ((chunk & QOI_MASK_2) == QOI_OP_DIFF) {
                px[0] += ((chunk >> 4) & 0x03) - 2;
                px[1] += ((chunk >> 2) & 0x03) - 2;
                px[2] += ( chunk       & 0x03) - 2;
            } else if ((chunk & QOI_MASK_2) == QOI_OP_LUMA) {
                int b2 = bytestream2_get_byteu(&gb);
                int vg = (chunk & 0x3f) - 32;
                px[0] += vg - 8 + ((b2 >> 4) & 0x0f);
                px[1] += vg;
                px[2] += vg - 8 +  (b2       & 0x0f);
            } else if ((chunk & QOI_MASK_2) == QOI_OP_RUN) {
                run = chunk & 0x3f;
            }

            memcpy(index[QOI_COLOR_HASH(px) & 63], px, 4);
        } else {
            break;
        }

        memcpy(&dst[off_x * 4], px, 4);
    }

    p->key_frame = 1;
    p->pict_type = AV_PICTURE_TYPE_I;

    *got_frame   = 1;

    return buf_size;
}

const FFCodec ff_qoi_decoder = {
    .p.name         = "qoi",
    .p.long_name    = NULL_IF_CONFIG_SMALL("QOI (Quite OK Image format) image"),
    .p.type         = AVMEDIA_TYPE_VIDEO,
    .p.id           = AV_CODEC_ID_QOI,
    .p.capabilities = AV_CODEC_CAP_DR1 | AV_CODEC_CAP_FRAME_THREADS,
    FF_CODEC_DECODE_CB(qoi_decode_frame),
};
