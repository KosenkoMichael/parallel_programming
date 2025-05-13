$blockSizes = @(
    @{ X = 8; Y = 8 },
    @{ X = 16; Y = 16 },
    @{ X = 32; Y = 32 }
)

$matMinSize = 100
$matMaxSize = 5100
$matMaxSizeGrow = 1000
$repeatCount = 5

foreach ($block in $blockSizes) {
    $threadsX = $block.X
    $threadsY = $block.Y

    for ($size = $matMinSize; $size -le $matMaxSize; $size += $matMaxSizeGrow) {
        for ($i = 0; $i -lt $repeatCount; $i++) {
            Write-Host "Running CUDA with matrix size $size, iteration $i, block $threadsX x $threadsY"
	    $combinedArg = "_$i" + "_" + "$threadsX"
            build\Release\cuda_matrix_mul.exe $size 0 100 $combinedArg $threadsX $threadsY
        }
    }
}