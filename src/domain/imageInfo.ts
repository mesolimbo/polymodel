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

  if (buf.length > 10 && buf.toString('ascii', 0, 3) === 'GIF') {
    return {
      mimeType: 'image/gif',
      extension: 'gif',
      width: buf.readUInt16LE(6),
      height: buf.readUInt16LE(8),
    };
  }

  return undefined;
}

export function toDataUrl(image: { bytes: Buffer; mimeType: string }): string {
  return `data:${image.mimeType};base64,${image.bytes.toString('base64')}`;
}

/** Pick the "w:h" preset closest to the requested shape, compared on a log scale so 2:1 and 1:2 miss 1:1 equally. */
export function nearestAspectRatio(width: number, height: number, presets: string[]): string {
  const distance = (preset: string) => {
    const [w, h] = preset.split(':').map(Number);
    return Math.abs(Math.log(w / h / (width / height)));
  };
  return presets.reduce((best, preset) => (distance(preset) < distance(best) ? preset : best));
}
