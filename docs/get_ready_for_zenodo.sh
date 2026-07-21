find outputs/outputs_scratch/ConFLAME_nrt-drivers3/ -type f -name "sample-prob*" -exec rm -v {} +

cd outputs/outputs_scratch/ConFLAME_nrt-drivers3/

for dir in */; do
    tar -cJf "${dir%/}.tar.xz" "$dir"
done
