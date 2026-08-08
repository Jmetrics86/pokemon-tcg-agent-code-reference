$treatments = @("02_aggressive", "03_defensive", "04_energy_hoarder", "05_fast_evolve", "06_boss_focus", "07_nonlinear_hp", "08_interaction_terms", "09_randomized_heuristics", "10_high_thresholds")

foreach ($t in $treatments) {
    Copy-Item "treatments\treatment_${t}.py" -Destination "main.py" -Force
    tar.exe -a -c -f submission.zip cg main.py deck.csv
    Write-Host "Submitting $t"
    $output = kaggle competitions submit -c pokemon-tcg-ai-battle -f submission.zip -m "Treatment $t" 2>&1
    Write-Host $output
    if ($output -match "0 submissions remaining") {
        Write-Host "Hit daily limit."
        break
    }
    Start-Sleep -Seconds 10
}
