# source extra/cl_android.sh
export LD_LIBRARY_PATH=/data/data/com.termux/files/usr/lib:/system/vendor/lib64
export LD_PRELOAD=/system/vendor/lib64/libOpenCL.so
export LIBC_PATH=/system/lib64/libc.so