{pkgs}: {
  deps = [
    pkgs.glibcLocales
    pkgs.xsimd
    pkgs.pkg-config
    pkgs.libxcrypt
    pkgs.ffmpeg-full
    pkgs.postgresql
    pkgs.openssl
  ];
}
