export interface SniffedImage {
  mimeType: string;
  extension: string;
  width?: number;
  height?: number;
}

/**
 * Identify the actual image format and dimensions from the bytes, since
 * providers sometimes mislabel the format they return.
 */
export function sniffImage(buf: Buffer): SniffedImage | undefined {
  if (buf.length > 24 && buf.readUInt32BE(0) === 0x89504e47) {
    return {
      mimeType: 'image/png',
      extension: 'png',
      width: buf.readUInt32BE(16),
      height: buf.readUInt32BE(20),
    };
  }

  if (buf.length > 4 && buf.readUInt16BE(0) === 0xffd8) {
    let pos = 2;
    while (pos + 9 < buf.length) {
      if (buf[pos] !== 0xff) {
        pos++;
        continue;
      }
      const marker = buf[pos + 1];
      // SOF0-SOF15 hold dimensions, except DHT (C4), JPG (C8), DAC (CC).
      if (marker >= 0xc0 && marker <= 0xcf && ![0xc4, 0xc8, 0xcc].includes(marker)) {
        return {
          mimeType: 'image/jpeg',
          extension: 'jpg',
          height: buf.readUInt16BE(pos + 5),
          width: buf.readUInt16BE(pos + 7),
        };
      }
      pos += 2 + buf.readUInt16BE(pos + 2);
    }
    return { mimeType: 'image/jpeg', extension: 'jpg' };
  }

  if (buf.length > 12 && buf.toString('ascii', 0, 4) === 'RIFF' && buf.toString('ascii', 8, 12) === 'WEBP') {
    return { mimeType: 'image/webp', extension: 'webp' };
  }

  return undefined;
}
