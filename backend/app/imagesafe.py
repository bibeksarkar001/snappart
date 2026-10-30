"""JPEG frames as stored: metadata stripped without re-encoding (pixels unchanged). Dropped: APP1 (EXIF, GPS, XMP), APP3-APP13 and APP15
(maker notes, IPTC / Photoshop) and comments. Kept: APP0 (JFIF), APP2 (ICC colour profile) and APP14 (Adobe colour transform), which
decoders need to get the colours right. Note: EXIF orientation goes too, so a frame is used as the sensor captured it."""
DROP = {0xE1, *range(0xE3, 0xEE), 0xEF, 0xFE}
STANDALONE = {0x01, *range(0xD0, 0xD8)}                                # TEM, RST0-7: no length field


class BadJpeg(ValueError):
    pass


def strip_metadata(data: bytes) -> bytes:
    if data[:2] != b"\xff\xd8":
        raise BadJpeg("not a JPEG")
    out, i, n = [b"\xff\xd8"], 2, len(data)
    while i < n:
        if data[i] != 0xFF:
            raise BadJpeg("bad marker")
        while i < n and data[i] == 0xFF:                                # fill bytes
            i += 1
        if i >= n:
            raise BadJpeg("truncated")
        m = data[i]
        i += 1
        if m in STANDALONE:
            out.append(bytes((0xFF, m)))
            continue
        if m == 0xD9:                                                   # EOI before any scan
            raise BadJpeg("no image data")
        if i + 2 > n:
            raise BadJpeg("truncated")
        length = int.from_bytes(data[i:i + 2], "big")
        if length < 2 or i + length > n:
            raise BadJpeg("bad segment length")
        if m == 0xDA:                                                   # start of scan: the rest is entropy-coded data, kept as is
            out.append(b"\xff\xda" + data[i:])
            return b"".join(out)
        if m not in DROP:
            out.append(bytes((0xFF, m)) + data[i:i + length])
        i += length
    raise BadJpeg("no image data")
