param(
    [string]$Path = "Bao_cao_do_an_ML_v2.docx"
)

$ErrorActionPreference = "Stop"
$resolved = (Resolve-Path -Path $Path -ErrorAction SilentlyContinue)
if (-not $resolved) {
    Write-Error "Không tìm thấy $Path. Chạy: node scripts/build_report.mjs"
    exit 1
}
$full = $resolved.Path

try {
    $word = New-Object -ComObject Word.Application
} catch {
    Write-Output "SKIP: Word COM không có trên máy này; mở Word và bấm Ctrl+A rồi F9 để cập nhật mục lục."
    exit 0
}

$word.Visible = $false
$word.DisplayAlerts = 0

try {
    $document = $word.Documents.Open($full, $false, $false)
    $document.Fields.Update() | Out-Null
    foreach ($section in $document.Sections) {
        foreach ($header in $section.Headers) { $header.Range.Fields.Update() | Out-Null }
        foreach ($footer in $section.Footers) { $footer.Range.Fields.Update() | Out-Null }
    }
    $document.Repaginate()
    $pages = $document.ComputeStatistics(2)
    $words = $document.ComputeStatistics(0)
    $document.Save()
    $document.Close($false)
    Write-Output "PAGES=$pages"
    Write-Output "WORDS=$words"
} finally {
    $word.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
}
