/*
 * HVQM4 demuxer
 * Copyright (c) 2021 Paul B Mahol
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

#include "libavutil/intreadwrite.h"

#include "avformat.h"
#include "internal.h"

typedef struct HVQM4Context {
    uint32_t nb_blocks;
    uint32_t current_block;
    uint32_t current_block_size;
    int64_t current_block_offset;
} HVQM4Context;

static int hvqm4_probe(const AVProbeData *p)
{
    if (memcmp(p->buf, "HVQM4 1.3", 9) &&
        memcmp(p->buf, "HVQM4 1.5", 9))
        return 0;

    return AVPROBE_SCORE_MAX;
}

static int hvqm4_read_header(AVFormatContext *s)
{
    HVQM4Context *hvqm4 = s->priv_data;
    AVIOContext *pb = s->pb;
    AVStream *vst, *ast;
    uint32_t header_size, usec_per_frame;
    int audio_format;

    vst = avformat_new_stream(s, NULL);
    if (!vst)
        return AVERROR(ENOMEM);

    avio_skip(pb, 16);

    header_size = avio_rb32(pb);
    avio_skip(pb, 4);
    hvqm4->nb_blocks = avio_rb32(pb);
    avio_skip(pb, 8);
    usec_per_frame = avio_rb32(pb);
    avio_skip(pb, 12);

    vst->codecpar->codec_type = AVMEDIA_TYPE_VIDEO;
    vst->codecpar->codec_id   = AV_CODEC_ID_MOBICLIP;
    vst->codecpar->width      = avio_rb16(pb);
    vst->codecpar->height     = avio_rb16(pb);

    avio_skip(pb, 2);
    avio_skip(pb, 2);

    avpriv_set_pts_info(vst, 64, 25, 1);

    ast = avformat_new_stream(s, NULL);
    if (!ast)
        return AVERROR(ENOMEM);

    ast->codecpar->codec_type = AVMEDIA_TYPE_AUDIO;
    ast->codecpar->channels   = avio_r8(pb);
    ast->codecpar->bits_per_coded_sample = avio_r8(pb);
    audio_format              = avio_r8(pb);
    switch (audio_format) {
    case 0:
        ast->codecpar->codec_id = AV_CODEC_ID_ADPCM_IMA_HVQM4;
        break;
    case 1:
        ast->codecpar->codec_id = AV_CODEC_ID_PCM_S16LE;
        break;
    }
    avio_skip(pb, 1);
    ast->codecpar->sample_rate = avio_rb32(pb);

    avio_skip(pb, header_size - avio_tell(pb));

    return 0;
}

static int hvqm4_read_packet(AVFormatContext *s, AVPacket *pkt)
{
    HVQM4Context *hvqm4 = s->priv_data;
    AVIOContext *pb = s->pb;
    int media_type, frame_type, ret;
    int32_t size;
    int64_t pos;

    if (avio_feof(pb))
        return AVERROR_EOF;

    pos = avio_tell(pb);

    if (hvqm4->current_block_offset >= hvqm4->current_block_size) {
        if (hvqm4->current_block_size)
            hvqm4->current_block++;
        avio_skip(pb, 4);
        hvqm4->current_block_size = avio_rb32(pb);
        hvqm4->current_block_offset = 0;
        avio_skip(pb, 8);
        avio_skip(pb, 4);
    }

    media_type = avio_rb16(pb);
    frame_type = avio_rb16(pb);
    size = avio_rb32(pb);
    ret = av_new_packet(pkt, size + 2);
    if (ret < 0)
        return ret;

    AV_WB16(pkt->data, frame_type);
    ret = avio_read(pb, pkt->data + 2, size);
    if (ret < 0)
        return ret;

    hvqm4->current_block_offset += avio_tell(pb) - pos;
    pkt->pos = pos;
    pkt->stream_index = media_type ? 0 : 1;
    if ((frame_type == 0x10 && media_type == 1) ||
        media_type == 0)
        pkt->flags |= AV_PKT_FLAG_KEY;

    return ret;
}

const AVInputFormat ff_hvqm4_demuxer = {
    .name           = "hvqm4",
    .long_name      = NULL_IF_CONFIG_SMALL("HVQM4"),
    .priv_data_size = sizeof(HVQM4Context),
    .read_probe     = hvqm4_probe,
    .read_header    = hvqm4_read_header,
    .read_packet    = hvqm4_read_packet,
    .extensions     = "h4m",
    .flags          = AVFMT_GENERIC_INDEX,
};
