PROC IMPORT DATAFILE="/folders/myshortcuts/C_DRIVE/Users/tamas/Documents/Data/RK_spine_volume_DATA_UTF_1.csv"
	OUT=spine
	DBMS=csv
	REPLACE
;
RUN;

PROC SQL;
CREATE TABLE spine_freq1 AS
SELECT Treatment, Value, COUNT(Value) AS Freq
FROM work.spine
WHERE Treatment="gts5_6" OR Treatment="gts5_1"
GROUP BY Value, Treatment
;
QUIT;

PROC SQL;
CREATE TABLE spine_freq2 AS
SELECT Treatment, Value, COUNT(Value) AS Freq
FROM work.spine
WHERE Treatment="gts5_6" OR Treatment="gts5_5"
GROUP BY Value, Treatment
;
QUIT;

ODS GRAPHICS ON;
PROC NPAR1WAY DATA=work.spine_freq1 EDF
	ALPHA=0.025
	PLOTS=edfplot
;
	CLASS treatment;
	VAR Value;
	FREQ Freq;
RUN;
ODS GRAPHICS OFF;

ODS GRAPHICS ON;
PROC NPAR1WAY DATA=work.spine_freq2 EDF
	ALPHA=0.025
	PLOTS=edfplot
;
	CLASS treatment;
	VAR Value;
	FREQ Freq;
RUN;
ODS GRAPHICS OFF;

PROC IMPORT DATAFILE="/folders/myshortcuts/C_DRIVE/Users/tamas/Documents/Data/RK_spine_volume_DATA_UTF.csv"
	OUT=spine_discr
	DBMS=csv
	REPLACE
;
RUN;

PROC FREQ DATA=work.spine_discr;
TABLES gts5_6 gts5_1 gts5_5;
RUN;

PROC UNIVARIATE DATA=work.spine_discr;
	VAR gts5_6 gts5_1 gts5_5;
	HISTOGRAM gts5_6 gts5_1 gts5_5 / NORMAL;
RUN;

PROC SQL;
	UPDATE work.spine
	SET Value=	
	CASE
	WHEN Value<=0.2 THEN 1
/*	
	WHEN 0.2<Value<=0.4 THEN 2
	WHEN 0.4<Value<=0.6 THEN 3
	WHEN 0.6<Value<=0.8 THEN 4
	WHEN 0.8<Value THEN 5
	ELSE 6
	*/
	ELSE 2
	END;
	
CREATE TABLE spine_freq2 AS
SELECT Treatment, Value, COUNT(Value) AS Freq
FROM work.spine
GROUP BY Value, Treatment
;
QUIT;

ODS GRAPHICS ON;
TITLE "Dendritc spine volume distribution for small and big spines (threshold: 0.2 um3)";
PROC SGPLOT DATA=work.spine_freq2;
	SERIES x=Treatment y=Freq / Group=Value groupdisplay=cluster clusterwidth=0.5;
  xaxis type=discrete;
RUN;
ODS GRAPHICS OFF;