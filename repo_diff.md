# Repo Differences
    MARS vs Chulab-Signal_Analyzer

`MARS repo`
    
    D:\chu_lab\MARS

`Chulab-Signal_Analyzer`

    D:\chu_lab\Chulab-Signal_Analyzer

## Same

1. 都無將腦影像 Downsample 及軸向轉換的程式碼，詳細需等我後續將程式碼補上。
2. 都沒辦法在沒有 `annotation.tif` 的情況下分析。
3. 都沒辦法進行 colocalize 分析
4. 

## Diff

1. MARS 有細胞體積過濾，Chulab-Signal_Analyzer 無。
2. 兩者從 mask 抓細胞的方式不同（不太確定怎麼形容，但 Chulab-Signal_Analyzer 的是 local maxima，MARS 應該不是）
3. MARS 內有針對 Downsample 及軸向轉換的程式碼，另一個無。
4. MARS 可以把 3D 處理後的檔案轉回 mask，另一個不行>
5. MARS 可以用 `annotation.tif` 把特定腦區抓出來，但另一個不行。

## Pipeline Needed

### 有對位需求的腦

1. Downsample + 軸向轉換（沒記錯應該是 y 跟 z 互換，請 double check）
2. BIRDs 腦區對位（不在這個 Repo 需求內），將產出 `annotation.tif` 之檔案供對位。
3. 3D 分析及 filter 並產生位置檔案（MARS 內是 `.xml`，另一個好像是 `.json` 嗎）（我一定需要體積過濾，不過 voxel size 每個腦可能不同，需要可以統一設定，因為每種 biomarker 的過濾大小都不一樣。請你幫我評估哪個檔案格式比較好。）
4. 腦區細胞數兩統計（MARS 部分是跟上面分開，也就是 `asym.py`，另一個是統一做，但他們參考的檔案應該都一樣。）
5. 用 `annotation.tif` 把特定腦區抓出來。（`aba2roi.py`）
6. 把過濾後的細胞畫回 mask，MARS 用 `x2xz.py`，另一個無。

### 不需要對位的腦

1. 3D 分析及 filter 並產生位置檔案（MARS 內是 `.xml`，另一個好像是 `.json` 嗎）（我一定需要體積過濾，不過 voxel size 每個腦可能不同，需要可以統一設定，因為每種 biomarker 的過濾大小都不一樣。請你幫我評估哪個檔案格式比較好。）
2. 腦區細胞數兩統計（MARS 部分是跟上面分開，也就是 `asym.py`，另一個是統一做，但他們參考的檔案應該都一樣。）
3. 把過濾後的細胞畫回 mask，MARS 用 `x2xz.py`，另一個無。

### 獨立需求：Colocalize 分析

我有時會需要針對兩種或以上的 biomarker 進行共定位分析。在分析時最重要的是共定位的定義。過往我以比較小的 biomarker 取質心，並將其與較大的 biomarker 之原本的 mask 比較，若落在其 mask 裡面，就計入有共定位。不過這部分需要你幫我評估遷移性。

### 其他需求

1. 我會需要每次跑分析都要有 log 檔案紀錄每次跑的數據、設定等。
2. 凡事優先考量合理性以及可遷移性。