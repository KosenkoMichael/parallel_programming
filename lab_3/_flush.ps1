Remove-Item -Path "stats.csv" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "build" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path "result" -Recurse -Force -ErrorAction SilentlyContinue

exit