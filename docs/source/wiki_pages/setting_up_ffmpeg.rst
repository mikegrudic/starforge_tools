Based on this guide
http://dridini.blogspot.com/2016/05/install-ffmpeg-opencv-without-root.html
Before starting, create “~/bin” and add it to PATH if you don’t already
have it We will skip yasm/nasm and the encodings we don’t need (e.g.
vorbis) from the guide No need to add –disable-static flags and we add
–disable-x86asm or –disable-asm flags if needed
######################### Detailed list of commands to follow:
######################### #Make ~/bin directory and add to path first,
then make build directories mkdir ffmpeg_sources mkdir ffmpeg_build cd
~/ffmpeg_sources #Getting x264 and compiling it: wget
http://anduin.linuxfromscratch.org/BLFS/x264/x264-20200218.tar.xz tar xf
x264-20200218.tar.xz mv x264-20200218 ./x264 cd x264
PKG_CONFIG_PATH=“$HOME/ffmpeg_build/lib/pkgconfig” ./configure
–prefix=“$HOME/ffmpeg_build” –bindir=“$HOME/bin” –enable-shared
–disable-asm make make install make distclean #Getting x265 and
compiling it: cd ~/ffmpeg_sources wget
http://ftp.videolan.org/pub/videolan/x265/x265_1.9.tar.gz tar xvf
x265_1.9.tar.gz mv x265_1.9 x265 cd ~/ffmpeg_sources/x265/build/linux
cmake -G “Unix Makefiles” -DCMAKE_INSTALL_PREFIX=“$HOME/ffmpeg_build”
-DENABLE_SHARED:bool=on -DDISABLE_STATIC:bool=off ../../source make make
install #Getting aac and compiling it: cd ~/ffmpeg_sources git clone
–depth 1 git://git.code.sf.net/p/opencore-amr/fdk-aac cd fdk-aac
autoreconf -fiv ./configure –prefix=“$HOME/ffmpeg_build” –enable-shared
–disable-asm make make install make distclean #Getting ffmpeg and
compiling it: export
LD_LIBRARY_PATH=$HOME/ffmpeg_build/lib/:$LD_LIBRARY_PATH export
PKG_CONFIG_PATH=$HOME/ffmpeg_build/lib/pkgconfig/:$PKG_CONFIG_PATH #Also
add both of the above lines to your .bashrc cd ~/ffmpeg_sources git
clone http://source.ffmpeg.org/git/ffmpeg.git cd ffmpeg ./configure
–prefix=“$HOME/ffmpeg_build”
–extra-cflags=“-I$HOME/ffmpeg_build/include”
–extra-ldflags=“-L$HOME/ffmpeg_build/lib” –bindir=“$HOME/bin”
–enable-gpl –enable-nonfree –enable-libfdk-aac –enable-libx264
–enable-pic –enable-shared –disable-x86asm make make install make
distclean hash -r #ffmpeg should now work with the command line
arguments in Sinkvis