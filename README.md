# Markowitz-Optimierung mit Out-of-Sample-Test

Optimale Portfolios nach Markowitz (Mean-Variance) aus zehn ETFs, geschlossen hergeleitet und numerisch mit Gewichtsgrenzen berechnet. Kern des Projekts ist ein Test, der die Optimierung auf Daten prüft, die sie nie gesehen hat.

## Fragestellung

Optimierte Portfolios sehen auf den Daten, mit denen sie berechnet wurden, immer gut aus. Die Frage ist, ob sie auch danach besser sind als ein gleich gewichtetes Portfolio (1/N).

## Daten

- Quelle: Yahoo Finance (`yfinance`), bereinigte Schlusskurse (Dividenden und Splits eingerechnet)
- **Daten bis zum 14.08.2026**, fester Stichtag in `src/config.py`, lokal gespeichert in `data/`
- 16.08.2016 bis 14.08.2026, 2.513 Handelstage
- ETFs: SPY (US-Aktien groß), QQQ (Nasdaq 100), IWM (US-Aktien klein), EFA (Industrieländer ohne USA), EEM (Schwellenländer), TLT (US-Staatsanleihen lang), LQD (US-Unternehmensanleihen), GLD (Gold), VNQ (US-Immobilien), DBC (Rohstoffe)

## Methode

| Baustein | Umsetzung |
|---|---|
| Minimum-Varianz, geschlossen | Lagrange: w = Σ⁻¹·1 / (1'Σ⁻¹·1), Leerverkäufe erlaubt |
| Tangentialportfolio, geschlossen | w ∝ Σ⁻¹(μ − rf) |
| Long-only mit Obergrenze | numerisch (SLSQP), 0 ≤ wᵢ ≤ 30 % |
| Effizienter Rand | kleinste Varianz je Zielrendite |
| Schrumpfung | (1 − δ)Σ + δ·diag(Σ) mit δ = 0,2 |
| Statischer Test | Gewichte aus Daten bis 29.12.2023, danach festgehalten |
| Walk-Forward | Neuoptimierung alle 63 Handelstage mit den letzten 756 Tagen, ab 19.08.2019 |
| Vergleich | 1/N und 5.000 zufällige Portfolios mit derselben Obergrenze |
## Ergebnisse

Zeitraum: Training 16.08.2016 bis 29.12.2023 (1.856 Handelstage), Test 02.01.2024 bis 14.08.2026 (657 Handelstage). Gewichte höchstens 30 % je ETF, risikofreier Zins 3 %.

**Out-of-Sample (statischer Test):**

| Strategie | Rendite p. a. | Volatilität p. a. | Sharpe Ratio | Max. Drawdown |
|---|---|---|---|---|
| 1/N | 17,2 % | 11,2 % | 1,27 | −11,5 % |
| Min-Varianz | 12,4 % | 8,2 % | 1,14 | −6,0 % |
| Max-Sharpe | 26,7 % | 14,2 % | 1,68 | −13,5 % |
| Max-Sharpe (Sigma geschrumpft) | 26,7 % | 14,1 % | 1,68 | −13,5 % |

- **Statischer Test:** Max-Sharpe liegt vor 1/N und besser als 4.999 von 5.000 Zufallsportfolios mit derselben Gewichtsgrenze (Rang 0,9998). 1/N liegt auf Rang 0,61, Min-Varianz auf 0,32.
- **Erklärung:** Max-Sharpe war im Training auf QQQ, GLD und SPY konzentriert, die im Test zu den stärksten ETFs gehörten (GLD 32,9 %, QQQ 25,5 %, SPY 22,1 % p. a.). Min-Varianz hielt viel TLT und LQD, die im Test schwach waren (−2,9 % und 2,9 % p. a.).
- **Walk-Forward:** Max-Sharpe hat auch hier die höchste Sharpe Ratio (0,73 gegen 0,60 bei 1/N), der Vorsprung ist aber deutlich kleiner. Min-Varianz liegt mit 0,51 unter 1/N.
- **Schrumpfung:** Die Kovarianzschrumpfung (δ = 0,2) senkt die Kondition von 202 auf 43, verbessert das Ergebnis aber nicht (Walk-Forward: 0,69 gegen 0,73).
- **Training gegen Test:** Alle Strategien haben im Test eine höhere Sharpe Ratio als im Training. Der Abstand von Max-Sharpe zu 1/N beträgt 0,35 im Training und 0,41 im Test.

Die Aussagen gelten für diesen Zeitraum und diese zehn ETFs, nicht allgemein.

![Effizienter Rand](output/effizienter_rand.png)
![Wertentwicklung im Testzeitraum](output/wertentwicklung_test.png)
![Walk-Forward](output/wertentwicklung_walk_forward.png)
![Rang gegen Zufallsportfolios](output/zufallsvergleich.png)

Im Diagramm zur Wertentwicklung im Testzeitraum liegt die Linie von Max-Sharpe unter der von Max-Sharpe (Sigma geschrumpft), weil beide fast identisch sind.

## Einschränkungen

- Ein einziger Zeitraum und zehn Titel, keine allgemeine Aussage über Markowitz.
- Der Testzeitraum war für alle Strategien günstiger als der Trainingszeitraum (1/N: Sharpe 0,35 im Training, 1,27 im Test).
- Erwartete Renditen sind Mittelwerte der Vergangenheit und damit stark fehlerbehaftet.
- Konstante Gewichte durch tägliches Rebalancing, keine Transaktionskosten, keine Steuern.
- Risikofreier Zins (3 %) und Gewichtsgrenze (30 %) sind Annahmen und wirken direkt auf die Ergebnisse.
- Keine Anlageberatung.

## Projekt ausführen

Python 3.10 oder neuer. Pakete installieren mit `pip install -r requirements.txt`, Analyse starten mit `python analyze.py`, Tests mit `python -m pytest` (25 Tests). Die Kursdaten liegen in `data/`, es ist kein Internet nötig.