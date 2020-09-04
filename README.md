# NLP - NAV
Henvendelseinnsikt fra nav.no til team personbruker 
skriv om oppgaven
## Topic modeling
Topic modeling er en mye brukt teknikk innen nlp. Topic modeling beskriver modeller som, i hovedsak v.ha unsupervised learning, fordeler tekster til forskjellige topics/kategorier. Dette brukes for å avdekke ustrukturerte sammenhenger i et tekstkorpus, og slikt vi har erfart - få rask og god innsikt i tekstkorpuset. I løpet av sommeren har vi brukt topic modeling til å få et bilde på hva nav.no brukes til.
Fordelen med topic modeling, og unsupervised learning generelt, er lettvintheten; uten for mye forhåndsarbeid kan man få god innsikt i tekstkorpuset.
Ulempen derimot er nokså åpenbar, nemlig at resultatene ikke nødvendigvis er de mest pålitelige. Pålitelige i den grad at tekstene i en topic ikke nødvendigvis har noen sammenheng. Denne ulempen er derimot ikke så viktig, hvis man tar topic modeling for det det er; et verktøy for å få innsikt om hva tekstene i et tekstcorpus handler om. 


se https://github.com/navikt/henvendelsesinnsikt-personbruker
## Tekstklassifisering
Et annet stort område innen nlp er tekstklassifisering. Tekstklassifisering handler om å fordele tekster til kategorier, men denne gangen, til forskjell fra topic modeling, vil kategoriene være bestemt på forhånd, og ofte (les alltid) trenes modellen opp på et annotert corpus (fasit). Dvs at modellen lærer seg selv mønstre for hvilke tekster som tilhører hvilke kategorier ved hjelp av en fasit, og så bruker denne erfaringen på det man kaller "unseen" tekster. 
Fordelen med tekstklassifisering er at man potensielt får mye mer pålitelige resultater, dvs. at tekstene ofte faktisk handler om de temaene/kategoriene de blir klassifisert til.
Ulempen er igjen ganske åpenbar. Det kreves her en større forkunnskap om tekstkorpuset og det kreves betydelig mer forarbeid fordi tekstene må annoteres (gis en fasit). 

Det er ved en kombinasjon av disse to områdene at man kanskje får den mest komplette analysen. Topic modeling hjelper med å få innblikk i tekstkorpuset og bør brukes til å velge kategorier/temaer som ikke overlapper, og som dekker mesteparten av tekstkorpuset. Dette er viktig for tekstklassifiseringsmodellene for å få modeller som ikke overfitter (dvs. at mønsteret modellen lærer er veldig spesifikt for treningsdataen)

## DistilBERT
Utvikling av norges første DistilBert (NLP)
se link til her

