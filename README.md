## `interface/tensorfuc.py`

此組件的核心是 `TensorFunction` 類別，它扮演著將使用者定義的任意 Python 函數與 TCI (Tensor Cross Interpolation) 演算法連接起來的關鍵橋樑。

### 物件與功能論述

  * **`TensorFunction` 物件**:
      * **核心用途**: 此物件的主要功能是作為一個「包裝器」(Wrapper)。它接收一個標準的 Python 函數，並賦予它一個類似高維張量的介面。在 TCI 演算法的視角中，它不需要知道函數內部的複雜邏輯；它只需要將這個 `TensorFunction` 物件當成一個可以依據索引 `σ` 來查詢其對應值 `F_σ` 的「黑盒子」。

      * **論文連結**: 論文中多次強調，TCI 演算法是透過對張量進行取樣來「學習」其結構 。`TensorFunction` 物件正是實現此概念的基礎。演算法透過呼叫此物件來取得張量元素的值，而無需存取整個（可能極其龐大的）張量 。

      * **關鍵屬性與行為**:

          * **`function`**: 這是 `TensorFunction` 在初始化時儲存的使用者定義函數。
          * **`cache`**: 這是一個字典或類似的雜湊表結構，用於儲存已經計算過的函數結果。其鍵(key)是輸入的索引（通常是元組），值(value)則是對應的函數計算結果。
          * **`__call__(indices)`**: 這是該物件最重要的呼叫方式。當 TCI 演算法需要某個索引 `indices` 的值時，它會呼叫 `tensor_func(indices)`。此方法會先檢查 `cache` 中是否存在該索引的結果。
              * 如果**存在**（快取命中），則直接回傳快取中的值，並增加 `cache_hits` 計數。
              * 如果**不存在**（快取未命中），它會呼叫儲存的 `function` 進行實際計算，將結果存入 `cache`，然後回傳結果。同時，`cache_size` 會增加。
          * **`cache_hits` 與 `cache_size`**: 這兩個屬性可以用於分析 TCI 演算法的效能，了解快取機制的效率以及演算法探索新索引的頻率。

### 如何呼叫與使用

使用者通常在 TCI 演算法開始前，先將自己的目標函數（例如，一個物理模型的能量函數或一個複雜的數學表達式）實例化為一個 `TensorFunction` 物件。

1.  **實例化**: 建立一個 `TensorFunction` 的實例，並將目標 Python 函數作為參數傳入其建構函式。

    ```python
    # 假設有一個計算成本高昂的函數 my_expensive_function
    tensor_func = TensorFunction(my_expensive_function)
    ```

2.  **呼叫**: TCI 演算法的內部邏輯會透過傳遞一個索引（通常是一個元組或列表）來呼叫此物件，以獲得張量在該位置的值。

    ```python
    # TCI 演算法內部需要索引 (3, 1, 4) 的值
    value = tensor_func((3, 1, 4))
    ```

### 概念驗證測試

要驗證 `TensorFunction` 的功能，重點在於確認其**快取機制**是否有效運作。測試流程如下：

1.  **定義一個可追蹤的函數**: 建立一個函數，內部包含一個計數器，每次被實際執行時，計數器加一。
2.  **首次呼叫**: 將此函數包裝成 `TensorFunction` 並用一組特定索引呼叫它。此時，應觀察到計數器變為 1，代表原始函數被執行了。
3.  **重複呼叫**: 使用**相同的索引**再次呼叫 `TensorFunction` 物件。此時，應觀察到計數器**維持在 1 不變**，因為結果是從快取中讀取的，並未再次執行原始函數。同時，可以檢查 `cache_hits` 屬性是否變為 1。
4.  **新索引呼叫**: 使用一組**新的索引**呼叫 `TensorFunction` 物件。應觀察到計數器變為 2，代表這次呼叫因為快取未命中而再次執行了原始函數。

這個驗證流程直接證明了 `TensorFunction` 的核心價值：對於計算成本高昂的函數，它能顯著減少總計算量，這對於 TCI 演算法的整體效能至關重要 。


-----

## `interface/Qgrid.py`

此組件提供了 `QuanticsGrid` 類別，它是實現論文中 **量化張量表示法 (Quantics Representation)** 的核心工具 。Quantics 的思想是透過將一個連續的物理變數 `x` 映射到其二進位的「量化」表示 `σ`，從而讓 TCI 演算法能夠處理具有指數級解析度的函數 。

`QuanticsGrid` 的真正威力在於它作為一個**翻譯層**，無縫地橋接了**使用者定義的物理函數**和**TCI 演算法內部的索引系統**。

### 物件與功能論述

  * **`QuanticsGrid` 物件**:
      * **核心用途**: 此物件扮演著一個「座標系統翻譯官」的角色。它本身不儲存龐大的網格資料，而是儲存了**轉換規則**。這些規則定義了如何在以下三種表示法之間進行轉換：

        1.  **物理座標 (Physical Coordinates)**: 使用者函數 `f(x, y, ...)` 所理解的連續或離散座標。
        2.  **線性網格索引 (Linear Grid Index)**: 虛擬離散化網格上的一個整數索引 `(m_x, m_y, ...)`。
        3.  **量化位元串 (Quantics Bit String)**: TCI 演算法所操作的張量索引 `σ` 。

      * **論文連結**: `QuanticsGrid` 直接對應論文第 6.1 節「Definition」中的概念 。論文指出，一個函數 `f(x)` 可以被視為一個張量 `F_σ = f(x(σ))` 。`QuanticsGrid` 正是實現了 `x(σ)` 這個關鍵的轉換函式。TCI 演算法在學習過程中，會向 `TensorFunction` 傳遞 `σ` 索引；而 `TensorFunction` 內部則會利用 `QuanticsGrid` 將 `σ` **翻譯**回物理座標 `x`，然後才呼叫使用者真正的函數 `f(x)`。

      * **關鍵屬性與行為**:

          * **`dim`**, **`nBit`**, **`a`**, **`b`**: 這些參數定義了座標系統的維度、解析度 (`2^nBit` 個點) 和物理範圍 `[a, b)` 。
          * **`unfoldingscheme`**: 位元串的排列方式，如 `'interleaved'`（交錯式），這對於找到最低秩的張量鏈表示至關重要 。
          * **`quantics_to_origcoord(σ)`**: 這是與 `TensorFunction` 互動時最核心的方法。它接收 TCI 演算法傳來的 `σ` 索引，並將其翻譯成使用者函數可以理解的物理座標。

### 如何呼叫與使用：結合 `TensorFunction`

`QuanticsGrid` 和 `TensorFunction` 的結合使用是執行 Quantics TCI 的標準模式。使用者並非將物理函數直接傳給 `TensorFunction`，而是傳入一個**包含了座標翻譯步驟**的新函數。

1.  **定義物理函數**: 首先，定義您想要分析的、以物理座標為輸入的函數。

    ```python
    # 物理函數 f(x, y)
    def my_physical_function(coords):
        x, y = coords
        return np.cos(x) * np.sin(y)
    ```

2.  **建立 `QuanticsGrid`**: 建立一個 `QuanticsGrid` 物件來定義座標系統。

    ```python
    from comphy_project.interface.Qgrid import QuanticsGrid

    # 建立一個 2 維，範圍 [-π, π)，每個維度用 10 個位元表示的網格
    grid = QuanticsGrid(dim=2, nBit=10, a=-np.pi, b=np.pi)
    ```

3.  **建立「翻譯」函數**: 建立一個新的函數，它接收 TCI 的 `σ` 索引，利用 `grid` 物件進行翻譯，然後再呼叫物理函數。

    ```python
    def tci_target_function(sigma):
        # 步驟 1: TCI 傳入 sigma 索引
        # 步驟 2: 使用 grid 物件將 sigma 翻譯成物理座標 (x, y)
        physical_coords = grid.quantics_to_origcoord(sigma)
        
        # 步驟 3: 將翻譯後的物理座標傳入真正的使用者函數
        return my_physical_function(physical_coords)
    ```

4.  **將「翻譯」函數傳給 `TensorFunction`**: 最後，將這個包含了翻譯邏輯的 `tci_target_function` 傳遞給 `TensorFunction` 進行封裝。

    ```python
    from comphy_project.interface.tensorfuc import TensorFunction

    tensor_func = TensorFunction(tci_target_function)

    # 這個 tensor_func 現在可以被傳遞給 TensorCI 主演算法
    # tci_solver = TensorCI(tensor_func, ...)
    ```

### 概念驗證測試

要驗證 `QuanticsGrid` 與 `TensorFunction` 的協同工作，我們需要模擬 TCI 演算法的行為，確認整個資訊流是正確的。

1.  **建立完整的流程**: 按照上述「如何呼叫與使用」的 4 個步驟，建立 `my_physical_function`, `grid`, `tci_target_function` 和 `tensor_func`。
2.  **選擇一個 TCI 索引**: 選擇一個量化位元串 `σ`。我們可以從一個線性索引 `m` 開始，然後用 `grid` 將其轉換成 `σ`，以確保其有效性。例如，選擇 `m = (100, 200)`。
3.  **手動計算理論值**:
      * 首先，將 `m = (100, 200)` 手動計算出其對應的物理座標 `(x, y)`。
      * 然後，將 `(x, y)` 代入 `my_physical_function`，得到理論上的函數值。
4.  **透過 `TensorFunction` 呼叫**:
      * 使用 `m` 透過 `grid.grididx_to_quantics(m)` 得到 `σ`。
      * 呼叫 `tensor_func(σ)`，這會觸發 `tci_target_function` 內部的完整翻譯流程。
5.  **比對結果**: 驗證 `tensor_func(σ)` 的回傳值是否與手動計算的理論值完全相符。如果相符，則證明 `QuanticsGrid` 成功地作為一個翻譯層，讓 `TensorFunction` 能夠為 TCI 演算法提供一個基於 `σ` 索引的正確介面，同時在內部處理了到物理座標的複雜映射。


***

## `interface/matrix_interface.py`

此組件定義了 `MatrixInterface` 類別，它在 TCI (Tensor Cross Interpolation) 演算法中扮演著一個至關重要的內部角色：作為一個「適配器」(Adapter)。它的主要功能是將一個高維的 `TensorFunction` 物件的一個特定「切片」(slice)，呈現成一個標準的二維矩陣介面，以供底層的矩陣分解演算法（如 prrLU）使用。

### 物件與功能論述

* **`MatrixInterface` 物件**:
    * **核心用途**: TCI 演算法的精髓在於將一個複雜的高維張量分解問題，簡化為一系列對二維矩陣進行分解的子問題。`MatrixInterface` 正是實現這一簡化的關鍵。它接收一個高維張量（以 `TensorFunction` 的形式）以及兩組用來定義切片的「多重索引」(multi-indices)，並將這個高維切片抽象化，使其對於呼叫者來說，行為和外觀都像一個普通的二維矩陣。

    * **論文連結**: `MatrixInterface` 是論文中演算法描述的直接程式碼體現。例如，在論文的 4.3.1 節中，演算法需要將 `Π_l` 張量視為一個矩陣 `F(I_{l-1} × S_l, S_{l+1} × J_{l+2})` 來進行 `prrLU` 分解 。`MatrixInterface` 物件就是這個「視為矩陣」的操作。它的 `row_indices` 和 `col_indices` 分別對應論文中的複合索引 `I_{l-1} × S_l` 和 `S_{l+1} × J_{l+2}`。底層的矩陣分解演算法不需要處理複雜的多重索引，只需透過 `MatrixInterface` 提供的標準介面即可。

    * **關鍵屬性與行為**:
        * **`tensor_func`**: 儲存了對應高維張量的 `TensorFunction` 物件。
        * **`row_indices`** 和 **`col_indices`**: 這兩個列表儲存了定義矩陣視圖的行與列的「多重索引」。列表中的每一個元素本身都是一個元組(tuple)，代表高維空間中的一組索引。
        * **`shape`**: 一個元組，表示這個二維矩陣視圖的維度（行數, 列數），其值由 `len(row_indices)` 和 `len(col_indices)` 決定。
        * **`get_value(row_idx, col_idx)`**: 這是此類別的核心方法。當底層的矩陣分解演算法需要矩陣中 `(row_idx, col_idx)` 位置的元素值時，它會呼叫此方法。此方法的內部邏輯是：
            1.  使用 `row_idx` 在 `row_indices` 列表中找到對應的行多重索引。
            2.  使用 `col_idx` 在 `col_indices` 列表中找到對應的列多重索引。
            3.  將這兩組多重索引合併，構成一個完整的「全局索引」。
            4.  將此全局索引傳遞給 `tensor_func` 物件進行查詢，並回傳結果。

### 如何呼叫與使用

`MatrixInterface` 是一個**內部組件**，使用者通常**不會直接與它互動**。它是在 `tensor_ci.py` 中的 TCI 演算法主循環（例如 `_update_bond` 方法）中被動態建立和使用的。

其內部使用流程大致如下：

1.  在 TCI 的某次迭代中，演算法決定需要對某個「鍵結」(bond) 進行更新。
2.  演算法會從 `IndexSet` 物件中獲取定義當前矩陣切片所需的行與列的多重索引。
3.  一個 `MatrixInterface` 物件被實例化，將 `TensorFunction` 和這兩組多重索引包裝起來。
4.  這個 `MatrixInterface` 物件隨後被傳遞給 `prrLU` 分解器。分解器完全透過 `get_value` 方法與張量互動，而無需關心其高維的本質。

### 概念驗證測試

要驗證 `MatrixInterface` 是否正確地履行了其「切片適配器」的職責，我們可以進行如下的概念驗證：

1.  **定義一個高維張量**: 首先，建立一個 `TensorFunction` 來代表一個已知的高維張量，例如一個 4 維的 `F(i,j,k,l)`，其函數值由一個簡單的、可預測的公式決定。
2.  **定義切片規則**: 建立兩組多重索引列表，例如 `row_indices` 由 `(i,j)` 組成，`col_indices` 由 `(k,l)` 組成。
3.  **建立矩陣視圖**: 使用上述的 `TensorFunction` 和索引列表來實例化一個 `MatrixInterface` 物件。
4.  **手動計算理論值**: 任意選擇一個二維矩陣的座標，例如 `(row=A, col=B)`。
    * 從 `row_indices[A]` 中找出對應的多重索引 `(i, j)`。
    * 從 `col_indices[B]` 中找出對應的多重索引 `(k, l)`。
    * 將它們合併成全局索引 `(i, j, k, l)`，並手動計算出 `F(i,j,k,l)` 的理論值。
5.  **實際呼叫比對**: 呼叫 `matrix_interface.get_value(A, B)`，取得實際的返回值。
6.  **驗證**: 比對理論值與實際返回值是否完全相等。如果相等，則證明 `MatrixInterface` 成功且正確地將高維索引的映射和查詢邏輯封裝起來，為底層演算法提供了可靠的二維矩陣抽象。

***

## `matrix/AdaptiveLU.py`

此組件包含了 **部分秩揭示 LU 分解 (Partial Rank-Revealing LU decomposition, prrLU)** 的核心實作，這是本專案中 TCI (Tensor Cross Interpolation) 演算法相較於傳統方法在**數值穩定性**上取得突破的關鍵。

### 物件與功能論述

* **`AdaptiveLU` (或其內部函式 `prrLU_solver`)**:
    * **核心用途**: 此物件或函式的主要職責是對一個給定的二維矩陣（由 `MatrixInterface` 提供）執行 `prrLU` 分解。它並非計算一個完整的 LU 分解，而是執行一個**迭代式**和**部分性**的分解過程，其目標是找到該矩陣最重要的「秩-`k`」近似。

    * **論文連結**: 這個組件是論文第 3.3 節「Partial rank-revealing LU decomposition」和 3.3.1 節「Default full search prrLU algorithm」 的直接程式碼實現。論文中詳細描述了 `prrLU` 的優勢與演算法流程：
        1.  **秩揭示 (Rank-Revealing)**: 演算法在每一步都會在剩餘的矩陣（即舒爾補, Schur complement）中尋找絕對值最大的元素作為下一個「樞紐」(pivot)。這種「完全樞軸」(full pivoting) 策略能有效地揭示矩陣的數值秩。
        2. **部分性 (Partial)**: 分解過程會在達到使用者指定的目標秩 `k` 或樞軸元素小於某個容忍度 `tolerance` 時提前終止 。這意味著它只計算了最重要的部分，而忽略了數值上不重要的部分，從而實現了近似與壓縮。
        3.  **數值穩定性**: 論文強調，`prrLU` 分解在數學上等價於交叉插值 (CI)，但數值上更為穩定，因為它**避免了直接建構和反轉可能病態 (ill-conditioned) 的樞紐矩陣 `P`** 。`AdaptiveLU.py` 正是透過迭代式的消去法來實現這一點。

    * **關鍵屬性與行為**:
        * **輸入**: 它接收一個 `MatrixInterface` 物件以及一個收斂容忍度 `tolerance` 作為輸入。
        * **迭代過程**:
            * 在第 1 步，它會在整個矩陣視圖中尋找絕對值最大的元素 `(i₁, j₁)`。
            * 這個元素被選為第一個樞紐。演算法接著執行一次高斯消去步驟，計算出舒爾補矩陣，即 `[A/A₁₁]`。
            * 在第 `r` 步，它會在第 `r-1` 步的舒爾補矩陣中尋找最大元素 `(i_r, j_r)` 作為第 `r` 個樞紐，並繼續此過程。
        * **輸出**: 其計算結果通常是一個包含以下資訊的資料結構 (e.g., a data class):
            * **`pivots`**: 一個列表，包含了被選中的 `k` 個樞紐的行列索引 `[(i₁, j₁), (i₂, j₂), ...]`。
            * **`L_factors`, `U_factors`, `D_factors`**: 分解出的 L (下三角)、U (上三角) 和 D (對角) 矩陣的相關因子。
            * **`rank`**: 最終分解得到的數值秩 `k`。

### 如何呼叫與使用

與 `MatrixInterface` 類似，`AdaptiveLU` 是一個**內部核心演算法**，使用者不會直接呼叫它。它被更高層的 TCI 演算法（在 `tensor_ci.py` 中）所呼叫。

其在 TCI 演算法中的角色如下：

1.  TCI 演算法在 `_update_bond` 階段，建立了一個代表當前矩陣切片的 `MatrixInterface` 物件。
2.  TCI 演算法將這個 `MatrixInterface` 物件傳遞給 `AdaptiveLU` (或 `prrLU_solver`)。
3.  `AdaptiveLU` 執行迭代式的 `prrLU` 分解，找到一組能很好地近似該矩陣切片的**最佳樞紐 (pivots)**。
4.  `AdaptiveLU` 將這組最佳樞紐回傳給 TCI 演算法。
5.  TCI 演算法使用這組新的、更優的樞紐來更新其 `IndexSet`，從而完成對該鍵結的優化，並進入下一個鍵結的更新。

### 概念驗證測試

要驗證 `AdaptiveLU` 的功能，重點在於確認它是否能**正確識別矩陣的秩**並**找到最大元素作為樞紐**。

1.  **建立一個已知秩的矩陣**: 首先，手動建立一個低秩矩陣。例如，一個 4x4 的秩-2 矩陣。可以讓矩陣中的某個元素值特別大，以方便預測樞紐的選擇。
    * 例如：`M = A * B`，其中 `A` 是 4x2 矩陣，`B` 是 2x4 矩陣。

2.  **設定目標**: 將這個矩陣包裝成一個 `MatrixInterface`（或直接使用 NumPy 陣列，如果函式支援），然後呼叫 `AdaptiveLU` 分解器，設定一個適當的 `tolerance`。

3.  **預測樞紐和秩**:
    * **第一次樞紐**: 理論上，分解器應該選擇矩陣中絕對值最大的元素作為第一個樞紐 `(i₁, j₁)`。
    * **第二次樞紐**: 在進行一次高斯消去後，分解器應該在剩餘的舒爾補矩陣中再次找到絕對值最大的元素作為第二個樞紐 `(i₂, j₂)`。
    * **終止與秩**: 由於原始矩陣是秩-2 的，在兩次迭代後，舒爾補矩陣中的所有元素值應該都非常接近於零（小於 `tolerance`）。因此，演算法應該終止，並回傳 `rank=2`。

4.  **實際呼叫與比對**: 執行 `AdaptiveLU` 分解，並檢查其回傳的 `rank` 和 `pivots` 列表。
    * 驗證回傳的 `rank` 是否確實為 2。
    * 驗證回傳的 `pivots` 列表中的前兩個樞紐是否與手動預測的樞紐位置一致。

這個流程驗證了 `AdaptiveLU` 成功地實現了論文中描述的「秩揭示」和「完全樞軸」策略，確保了 TCI 演算法在選擇最重要的資訊點時的準確性和高效性。

***

## `matrix/crossdata.py`

此組件定義了 `CrossData` 類別，它是一個專門用來儲存矩陣分解結果的**資料容器 (Data Container)**。在 TCI 演算法的流程中，當 `AdaptiveLU` (prrLU 分解器) 完成對一個矩陣切片的分析後，會將所有重要的計算結果封裝到一個 `CrossData` 物件中並回傳。

### 物件與功能論述

* **`CrossData` 物件**:
    * **核心用途**: 它的主要功能是作為一個結構化的資料載體，負責從底層的矩陣分解演算法 (`AdaptiveLU.py`) 向高層的 TCI 主演算法 (`tensor_ci.py`) 傳遞資訊。它將 `prrLU` 分解產生的多個不同部分（如樞紐、秩、L/U 因子等）整合成一個單一、清晰的物件，簡化了函式之間的資料傳遞。

    * **論文連結**: `CrossData` 物件儲存的內容直接對應論文中 `prrLU` 分解的核心結果。具體來說，它儲存了論文中公式 (28) 和 (30) 所描述的各個組成部分：
        * `A ≈ LDU`
        `CrossData` 儲存了近似重建這個矩陣所需的所有元素：`L` 和 `U` 的因子、對角的 `D` 因子、以及最重要的，決定了這個近似結構的樞紐 (pivots)。它代表了對一個二維矩陣切片進行交叉插值 (Cross Interpolation, CI) 或 `prrLU` 分解後得到的完整資訊集。

    * **關鍵屬性**:
        * **`pivots`**: 一個列表，記錄了在分解過程中被選中的樞紐的**行列索引** `[(i₁, j₁), (i₂, j₂), ...]`。這是 `CrossData` 中最重要的資訊，因為它直接決定了 TCI 張量鏈的結構，TCI 演算法將用它來更新 `IndexSet`。
        * **`rank`**: 分解後得到的矩陣的**數值秩** `k`。這個值告訴 TCI 演算法在當前的近似下，需要多大的鍵結維度 (bond dimension)。
        * **`L_factors`, `U_factors`**: 儲存了分解後得到的下三角矩陣 `L` 和上三角矩陣 `U` 的非平凡部分。這些因子可以用來重建近似矩陣。
        * **`D_factors`** (或類似名稱): 儲存了對角矩陣 `D` 的元素，即樞紐元素的值。
        * **`tolerance`**: 記錄了此次分解所使用的收斂容忍度。

### 如何呼叫與使用

`CrossData` 是一個**被動的資料物件**，使用者或高層演算法不會主動「呼叫」它，而是「接收」並「讀取」它。

其在 TCI 演算法中的生命週期如下：

1.  **生成**: 在 `AdaptiveLU.py` 中的 `prrLU` 分解函式執行完畢後，該函式會將其計算結果（樞紐、秩等）打包，並**實例化**一個 `CrossData` 物件。
2.  **回傳**: `prrLU` 分解函式將這個填充好資料的 `CrossData` 物件**回傳**給呼叫它的 TCI 主演算法。
3.  **使用**: TCI 主演算法接收到 `CrossData` 物件後，會從中**讀取**所需的屬性。最主要的操作是讀取 `pivots` 屬性，用以更新全局的 `IndexSet`。它也可能會讀取 `rank` 屬性來監控收斂情況。

### 概念驗證測試

要驗證一個 `CrossData` 物件的內容是否正確，實際上是在驗證產生它的那個分解過程 (`AdaptiveLU`) 是否正確。其驗證流程如下：

1.  **準備一個測試矩陣**: 建立一個性質已知的測試矩陣，例如一個秩為 2 的 5x5 矩陣。
2.  **執行分解**: 呼叫 `AdaptiveLU` 分解器對該矩陣進行分解，這會產生一個 `CrossData` 物件。
3.  **檢查 `CrossData` 的屬性**:
    * **驗證秩**: 檢查 `cross_data.rank` 的值是否等於預期的秩（在此例中為 2）。
    * **驗證樞紐**: 檢查 `cross_data.pivots` 列表的長度是否等於秩。同時，根據測試矩陣的元素分佈，手動預測樞紐應該在哪些位置，並與 `pivots` 列表中的實際值進行比對。
    * **(可選) 驗證重建**: 使用 `CrossData` 中儲存的 `L`, `U`, `D` 因子，根據公式 `A ≈ LDU` 來手動重建近似矩陣。然後，計算這個重建矩陣與原始測試矩陣之間的誤差，驗證誤差是否在 `tolerance` 的範圍內。


***

### `matrix/cytnx_prrLU.py`

此組件是 `prrLU` (Partial Rank-Revealing LU decomposition) 演算法的**高效能運算後端**。它利用 `cytnx` 函式庫來執行密集的數值計算，確保 TCI (Tensor Cross Interpolation) 演算法在處理大規模矩陣切片時的效能與效率。

#### 物件與功能論述

* **`prrLU_solver` (或類似名稱的函式)**:
    * **核心用途**: 此函式是 `AdaptiveLU.py` 中定義的演算法邏輯的具體數值實現。它接收一個 `MatrixInterface` 物件，並使用 `cytnx` 提供的張量（Tensor）和線性代數（LinAlg）工具來執行迭代式的高斯消去和樞紐搜尋。

    * **論文連結**: 如果說 `AdaptiveLU.py` 是論文第 3.3.1 節演算法流程的「偽代碼」，那麼 `cytnx_prrLU.py` 就是將其轉化為高效能程式碼的「引擎」。`cytnx` 函式庫底層通常由 C++ 編寫並經過最佳化，能夠快速處理 `prrLU` 分解中涉及的大量矩陣元素存取和更新操作，特別是在計算舒爾補 (Schur complement) 的步驟中。這確保了演算法的實際執行效能，使其能夠應用於論文中提到的高維度問題。

    * **關鍵行為**:
        * **與 `cytnx` 的整合**: 此模組會將從 `MatrixInterface` 獲取的數值轉換為 `cytnx.Tensor` 物件，以利用 `cytnx` 的高效運算能力。
        * **樞紐搜尋 (Pivot Search)**: 它會實作在 `cytnx.Tensor` 上尋找絕對值最大元素的邏輯，這對應了論文中的「完全樞軸」(full pivoting) 策略。
        * **舒爾補計算**: 在選定樞紐後，它會利用 `cytnx` 的矩陣運算功能（如外積、矩陣乘法和減法）來高效地計算下一步迭代所需的舒爾補矩陣。
        * **結果封裝**: 計算完成後，它會將找到的樞紐、計算出的秩以及 L/U 因子等資訊，打包到一個 `CrossData` 物件中並回傳。

#### 如何呼叫與使用

`cytnx_prrLU.py` 是 TCI 專案的**最底層計算核心之一**，它不會被使用者直接呼叫，甚至不會被 `tensor_ci.py` 直接呼叫。它的呼叫者是更高一層的抽象，例如 `AdaptiveLU.py` 或 `mat_decomp.py`。

其在專案中的呼叫鏈如下：
`tensor_ci.py` -> `mat_decomp.py` (或 `AdaptiveLU.py`) -> `cytnx_prrLU.py`

TCI 主演算法將矩陣分解的任務委派給 `AdaptiveLU.py`，而 `AdaptiveLU.py` 則將實際的數值運算任務委派給 `cytnx_prrLU.py` 中的函式來執行。

#### 概念驗證測試

驗證 `cytnx_prrLU.py` 的重點在於確認這個高效能後端所產出的**結果**，與 `AdaptiveLU.py` 中描述的**演算法邏輯**是完全一致且準確的。

1.  **建立測試案例**: 準備一個已知秩和樞紐位置的測試矩陣（與 `AdaptiveLU.py` 的驗證案例相同）。
2.  **執行分解**: 透過高層介面（例如 `AdaptiveLU.py`）呼叫，確保最終是由 `cytnx_prrLU.py` 中的函式來執行分解。
3.  **比對結果**: 取得回傳的 `CrossData` 物件。
    * **驗證秩**: 檢查其 `rank` 屬性是否與矩陣的理論秩相符。
    * **驗證樞紐**: 檢查其 `pivots` 屬性是否與理論上應該被選中的最大元素位置相符。
    * **驗證數值精度**: 如果可能，使用 `CrossData` 中的 L/U 因子重建近似矩陣，並計算與原始矩陣的 Frobenius 範數誤差，確保誤差在容忍度 `tolerance` 之內。

這個流程確保了 `cytnx` 後端的實作是可靠的，並且其高效的計算能力沒有以犧牲準確性為代價。

***

### `matrix/mat_decomp.py`

此組件作為一個**矩陣分解的調度中心或高層 API**。它封裝了對不同矩陣分解策略（目前主要是 `prrLU`）的呼叫細節，為 TCI 主演算法提供了一個更簡潔、更統一的介面。

#### 物件與功能論述

* **`decompose_matrix` (或類似名稱的函式)**:
    * **核心用途**: 它的主要功能是作為一個**策略分派器 (Strategy Dispatcher)**。當 TCI 演算法需要分解一個矩陣切片時，它只需要呼叫這個高層函式，而不需要關心底層是用 `cytnx` 實現的 `prrLU`，還是未來可能加入的其他分解方法（例如，基於 SVD 的方法）。

    * **論文連結**: 雖然論文主要聚焦於 `prrLU`，但一個設計良好的軟體架構會考慮未來的擴充性。`mat_decomp.py` 提供了這樣的擴充點。它將「分解一個矩陣」這個**意圖**，與「如何分解一個矩陣」這個**實現**分離開來。這使得專案可以輕鬆地在不同的分解演算法之間進行切換和比較，這對於學術研究和演算法開發非常有價值。

    * **關鍵行為**:
        * **簡化 API**: 它為 TCI 主演算法提供了一個非常簡單的呼叫介面，可能只需要傳入 `MatrixInterface` 和 `tolerance` 兩個參數。
        * **呼叫後端**: 其內部邏輯會直接呼叫 `cytnx_prrLU.py` 中的 `prrLU_solver` 函式來執行實際的計算。
        * **統一回傳**: 無論底層使用哪種分解方法，它都會確保回傳的格式是統一的 `CrossData` 物件，從而讓 TCI 主演算法可以一致地處理分解結果。
        * **擴充性**: 如果未來需要加入 SVD-based 的分解方法，只需在此檔案中新增一個分支邏輯，而無需修改 TCI 主演算法的程式碼。

#### 如何呼叫與使用

`mat_decomp.py` 是 TCI 主演算法 (`tensor_ci.py`) 與底層數值計算 (`cytnx_prrLU.py`) 之間的**中間層**。

1.  **被呼叫**: TCI 主演算法在需要進行矩陣分解時，會呼叫 `mat_decomp.py` 中的主函式，例如 `decompose_matrix(matrix_view, tolerance)`。
2.  **執行調度**: `decompose_matrix` 函式內部會直接呼叫 `cytnx_prrLU.prrLU_solver`，並將參數透傳過去。
3.  **回傳結果**: 它接收來自 `prrLU_solver` 的 `CrossData` 物件，並將其原封不動地回傳給 TCI 主演算法。

#### 概念驗證測試

驗證 `mat_decomp.py` 的功能，主要是確保這個中間層的**參數傳遞**和**結果回傳**是正確無誤的。

1.  **準備測試案例**: 與前述相同，建立一個已知性質的測試矩陣，並將其包裝成 `MatrixInterface`。
2.  **呼叫高層 API**: 直接呼叫 `mat_decomp.py` 中的主函式，例如 `decompose_matrix`，並傳入測試物件和 `tolerance`。
3.  **驗證回傳物件**: 檢查回傳的 `CrossData` 物件。
    * 確認物件的 `rank` 和 `pivots` 屬性是否與預期相符。
    * 這個測試流程看似與 `cytnx_prrLU.py` 的測試重複，但其核心目的不同。這裡的重點是驗證 `mat_decomp.py` 作為一個「調度員」是否稱職，確保它能正確地將任務分派下去，並將結果完整地帶回來，而沒有在中間層發生任何資訊的遺失或錯誤的處理。

***

## `matrix/pivot_finder.py`

此組件負責 TCI (Tensor Cross Interpolation) 演算法中最核心的決策之一：**樞紐選擇 (Pivot Selection)**。它實作了在一個矩陣（或舒爾補）中尋找下一個最佳樞紐的策略。樞紐的選擇品質直接影響到 `prrLU` 分解的效率和 TCI 演算法的收斂速度。

### 物件與功能論述

* **`PivotFinder` (或其內部函式 `find_pivot_full`, `find_pivot_rook`)**:
    * **核心用途**: 此組件的主要功能是根據指定的策略，在一個二維矩陣視圖 (`MatrixInterface`) 中找到數值上最重要的元素，並回傳其索引。這個元素將被用作 `prrLU` 分解的下一步樞紐。

    * **論文連結**: `pivot_finder.py` 是論文第 3.3.2 節「Alternative pivot search methods: full, rook or block rook」的具體程式碼實現。論文中比較了不同的樞紐搜尋策略，而此組件正是這些策略的執行者：
        1.  **完全搜尋 (Full Search)**:
            * **對應論文**: 這是 `prrLU` 的預設策略 。
            * **行為**: 演算法會**遍歷**整個剩餘矩陣中的每一個元素，找到絕對值最大的那一個作為樞紐 。
            * **優缺點**: 這種方法最為穩健，能保證找到全局最佳的下一個樞紐，但計算成本也最高，其複雜度為 `O(mn)`，其中 `m` 和 `n` 是矩陣的維度 。
        2.  **Rook 搜尋 (Rook Search)**:
            * **對應論文**: 論文中提到這是一種計算成本更低的替代方案 。
            * **行為**: 此策略模擬西洋棋中的「城堡」(Rook) 移動方式。它從一個隨機位置開始，交替地在行和列上尋找最大值：找到當前行中的最大元素，然後跳到該元素的列，再尋找該列中的最大元素，如此反覆，直到找到一個同時是其所在行和所在列的最大值的元素（達到「Rook 條件」）。
            * **優缺點**: 其計算成本顯著降低至 `O(m+n)`，並且在實務上，其穩定性和收斂性與完全搜尋相當 。

    * **關鍵行為**:
        * **輸入**: 接收一個 `MatrixInterface` 物件，代表當前需要被搜尋的矩陣。
        * **輸出**: 回傳一個元組 `(row_idx, col_idx)`，代表找到的最佳樞紐在 `MatrixInterface` 中的二維索引。

### 如何呼叫與使用

`pivot_finder.py` 是 `prrLU` 分解器 (`cytnx_prrLU.py`) 的內部輔助工具，不會被使用者直接呼叫。

其在 `prrLU` 分解的迭代迴圈中的使用流程如下：

1.  `prrLU` 分解器在第 `k` 次迭代時，會擁有一個代表第 `k-1` 步舒爾補的矩陣視圖。
2.  分解器會呼叫 `pivot_finder.py` 中的函式（例如 `find_pivot_full`），並將該矩陣視圖作為參數傳入。
3.  `pivot_finder.py` 執行其搜尋邏輯，找到最佳樞紐的索引 `(i_k, j_k)` 並回傳。
4.  `prrLU` 分解器接收到這個索引後，將其記錄到 `CrossData` 的 `pivots` 列表中，並利用這個樞紐進行下一步的高斯消去，計算出第 `k` 步的舒爾補。

### 概念驗證測試

要驗證 `pivot_finder.py` 的功能，需要針對其支援的不同搜尋策略進行測試。

* **驗證 `Full Search`**:
    1.  **建立測試矩陣**: 建立一個矩陣，並在一個非角落的、預先設定好的位置 `(r, c)` 放置一個絕對值遠大於其他所有元素的數值。
    2.  **執行搜尋**: 呼叫 `find_pivot_full` 函式。
    3.  **比對結果**: 驗證回傳的樞紐索引是否**精確地**等於 `(r, c)`。這證明了完全搜尋策略的正確性。

* **驗證 `Rook Search`**:
    1.  **建立特殊測試矩陣**: 建立一個矩陣，其中絕對值最大的元素（全局最大值）位於 `(r1, c1)`，但存在另一個元素位於 `(r2, c2)`，它雖然不是全局最大值，但卻是其所在行和所在列的最大值（即滿足「Rook 條件」的鞍點）。
    2.  **執行搜尋**: 呼叫 `find_pivot_rook` 函式。
    3.  **比對結果**: 驗證回傳的樞紐索引是否為 `(r2, c2)` 而**不是** `(r1, c1)`。這證明了 Rook 搜尋演算法的獨特邏輯被正確實現，它尋找的是滿足局部最優條件的點，而非全局最大值。

這個流程確保了 `pivot_finder.py` 能夠根據指定的策略，準確地為 `prrLU` 分解提供核心的決策依據。


-----

## `tensor/auto_mpo.py`

此組件是論文中一個非常重要的應用實現：**自動化矩陣乘積算符 (Matrix Product Operator, MPO) 的建構**。在量子多體物理中，哈密頓量 (Hamiltonian) 通常可以表示為一系列局域算符乘積的和。`auto_mpo.py` 提供了一套工具，可以將這種符號化的算符和，自動且高效地壓縮成一個緊湊的 MPO 張量鏈。

### 物件與功能論述

此組件透過定義一系列階層化的類別來將一個複雜的哈密頓量符號化：

  * **`locOp` (Local Operator) 物件**:

      * **核心用途**: 代表作用在單一格點 (single site) 上的**局域算符**，本質上是一個小型的矩陣（例如，自旋-1/2 系統中的 2x2 泡利矩陣）。
      * **論文連結**: 這是構成 MPO 的最基本單位。論文中的公式 (91) 和 (93) 描述的哈密頓量，都是由像 `Sᶻ` 或 `c†` 這樣的局域算符所組成。

  * **`prodOp` (Product of Operators) 物件**:

      * **核心用途**: 代表哈密頓量中的一個**單項**，即一連串 `locOp` 的直積 (direct product)。例如，`Sᶻᵢ * Sᶻᵢ₊₁` 就是一個 `prodOp`。在張量網絡的語言中，每一個 `prodOp` 本身就是一個秩為 1 的 MPO。
      * **論文連結**: `prodOp` 直接對應論文公式 (87) 中的單一項 `H_a`。哈密頓量 `H` 就是由許多這樣的 `prodOp` 項相加而成。

  * **`polyOp` (Polynomial of Operators) 物件**:

      * **核心用途**: 代表**完整的哈密頓量**，即一系列 `prodOp` 物件的總和（多項式）。它讓使用者可以用非常直觀的方式，透過重載的 `+` 和 `*` 運算符，像寫數學公式一樣來建構複雜的哈密頓量。
      * **論文連結**: `polyOp` 物件就是論文公式 (87) 中的 `H = Σ H_a` 的程式碼表示。

  * **`to_tensorTrain()` 方法**:

      * **核心用途**: 這是 `polyOp` 物件最重要的方法。它執行一個**自動化的壓縮演算法**，將一個由大量（可能成千上萬）秩-1 MPO (`prodOp`) 組成的和，轉換成一個單一的、最佳化且低秩的 MPO 張量鏈。
      * **論文連結**: 此方法實現了論文第 7.2 節中描述的「**分而治之 (divide-and-conquer)**」策略。它並非一次性將所有項相加（這會導致 MPO 的鍵結維度爆炸性增長），而是分批次地將 `prodOp` 項相加並使用基於 `prrLU` 的 **CI-canonicalization** 進行壓縮，最後再將這些部分和合併起來。這種方法在處理量子化學中項數極多的哈密頓量時（如論文表 3 所示），相比傳統的 SVD 壓縮方法，具有顯著的**數值穩定性優勢**。

### 如何呼叫與使用

使用者可以透過組合 `locOp`, `prodOp`, 和 `polyOp` 來直觀地建構哈密頓量，然後呼叫 `to_tensorTrain()` 來生成 MPO。

以下以論文中的**海森堡模型 (Heisenberg Model)** 為例，展示其呼叫方式：

1.  **定義局域算符**: 首先，定義系統所需的基本 `locOp`，例如自旋的上旋 (`Sp`)、下旋 (`Sm`) 和 `Sz` 算符。

    ```python
    # 假設 Sz, Sp, Sm 已經被定義為 locOp 物件
    Sz = locOp(...)
    Sp = locOp(...)
    Sm = locOp(...)
    ```

2.  **建構 `polyOp` 物件**: 像寫數學公式一樣，將算符的乘積和相加，來建立一個 `polyOp` 物件 `H`。

    ```python
    L = 50 # 系統大小
    H = polyOp() # 初始化一個空的哈密頓量

    for i in range(L):
        # Sᶻᵢ * Sᶻᵢ₊₁ 項
        term1 = prodOp({i: Sz, (i+1)%L: Sz})
        
        # S⁺ᵢ * S⁻ᵢ₊₁ 項
        term2 = prodOp({i: Sp, (i+1)%L: Sm}) * 0.5
        
        # S⁻ᵢ * S⁺ᵢ₊₁ 項
        term3 = prodOp({i: Sm, (i+1)%L: Sp}) * 0.5

        H += term1 + term2 + term3
    ```

3.  **生成 MPO**: 在 `polyOp` 物件 `H` 上呼叫 `to_tensorTrain()` 方法，即可得到最終壓縮好的 MPO。

    ```python
    # tolerance 參數控制壓縮的精度
    mpo = H.to_tensorTrain(tolerance=1e-10)

    # mpo 現在是一個 TensorTrain 物件，可以直接用於 DMRG 等後續計算
    print(f"生成的 MPO 的最大鍵結維度: {mpo.max_bond_dimension}")
    ```

### 概念驗證測試

要驗證 `auto_mpo.py` 的功能，重點在於確認生成的 MPO 是否**準確地**代表了原始的、由多項式定義的哈密頓量。

1.  **建構哈密頓量**: 使用上述步驟建構一個 `polyOp` 物件 `H`。
2.  **生成 MPO**: 呼叫 `H.to_tensorTrain()` 產生 MPO。
3.  **計算內積 (Overlap)**: 論文的 Listing 4 和公式 (90) 提供了一種驗證方法。我們可以計算原始 `polyOp` `H` 與生成的 `mpo` 之間的內積 `<H|mpo>`，以及 `mpo` 自身的範數 `<mpo|mpo>`。
4.  **驗證**: 一個正確的 MPO 應該滿足 `|<H|mpo>|² / (<H|H> * <mpo|mpo>) ≈ 1`。在實作上，通常會提供一個類似 `H.overlap(mpo)` 的輔助函式來進行此項檢查。如果內積結果與範數相符，則證明 `auto_mpo` 成功地將符號化的哈密頓量轉換成了準確且緊湊的 MPO 表示，其結構與論文中預期的理論值（例如海森堡模型的鍵結維度為 8）一致。


-----

## `tensor/tensor_train.py`

此組件定義了 `TensorTrain` 類別，它是整個專案的**核心資料結構**。在 TCI (Tensor Cross Interpolation) 演算法完成對一個高維張量的學習與分解後，其最終的產出——一個矩陣乘積態 (Matrix Product State, MPS) 或張量鏈——就是以 `TensorTrain` 物件的形式存在的。此物件不僅儲存了張量鏈的結構，還提供了一系列對其進行操作和分析的基本方法。

### 物件與功能論述

  * **`TensorTrain` 物件**:
      * **核心用途**: 此物件的根本用途是**表示一個高維張量**的低秩近似形式。它將一個原本需要指數級儲存空間的張量 `F(σ₁, σ₂, ..., σ_L)`，表示為一系列小型三階張量（核心張量）的矩陣乘積。

      * **論文連結**: `TensorTrain` 物件是論文中反覆出現的核心概念的直接程式碼實體。論文公式 (1) `F_σ ≈ M₁ * M₂ * ... * M_L` 就是 `TensorTrain` 物件所代表的數學結構。論文中所有的 TCI 演算法，其最終目標都是建構或優化一個 `TensorTrain` 物件。

      * **關鍵屬性**:

          * **`cores`** (或 `_cores`)\*\*: 一個列表 (list)，儲存了組成張量鏈的所有核心張量 `[M₁, M₂, ..., M_L]`。列表中的每一個元素 `M_l` 都是一個三階張量，其維度為 `(χ_{l-1}, d_l, χ_l)`，分別對應輸入的鍵結維度 (bond dimension)、物理維度 (physical dimension) 和輸出的鍵結維度。
          * **`shape`**: 描述了此張量鏈所代表的高維張量的形狀，例如 `(d₁, d₂, ..., d_L)`。
          * **`bond_dimensions`**: 儲存了所有內部鍵結的維度 `[χ₀, χ₁, ..., χ_L]`。`max(bond_dimensions)` 即為此張量鏈的秩 (rank)。

      * **關鍵方法**:

          * **`eval(sigma)`**:
              * **功能**: 計算張量鏈在給定索引 `sigma = (σ₁, σ₂, ..., σ_L)` 上的具體數值 `F_σ`。
              * **實現**: 透過將一系列核心張量 `M_l` 依序進行矩陣乘法來實現。`eval` 的效率遠高於直接儲存和查詢一個巨大的多維陣列。
          * **`sum()`**:
              * **功能**: 計算張量所有元素的總和 `Σ_σ F_σ`。這在論文應用中尤其重要，例如計算高維積分或物理系統的配分函數（見論文第 5 節）。
              * **實現**: 透過對每個核心張量 `M_l` 的物理維度進行求和，然後將結果（一系列矩陣）相乘來高效完成，避免了指數級的求和運算。
          * **`compressCI()`**, **`compressSVD()`**, **`compressLU()`**:
              * **功能**: 對當前的 `TensorTrain` 進行再壓縮，以達到更低的秩或更高的精度。
              * **論文連結**: 這些方法是論文第 4.5 節「CI- and LU-canonicalization」中討論的演算法的實現。`compressCI` 尤其重要，因為它利用 TCI 的思想來壓縮一個已有的張量鏈，這在 `auto_mpo` 等應用中至關重要。`compressSVD` 則是更傳統的、基於奇異值分解的壓縮方法，可作為比較基準。
          * **運算符重載 (如 `+`, `*`)**:
              * **功能**: 允許使用者以直觀的方式對兩個 `TensorTrain` 物件進行元素級的加法或乘法。
              * **論文連結**: 這對應論文第 4.7 節「Operations on tensor trains」。例如，在 `auto_mpo` 中，將哈密頓量的不同項相加，就是透過 `+` 運算符將它們的 MPO（一種 `TensorTrain`）相加，然後再進行壓縮。

### 如何呼叫與使用

`TensorTrain` 物件通常是 TCI 演算法的**輸出**，但使用者也可以手動建立或操作它。

1.  **獲取物件**: 最常見的方式是從一個 `TensorCI` 物件中獲取計算結果。

    ```python
    # 假設 tci_object 是一個已經收斂的 TensorCI 實例
    my_tensor_train = tci_object.get_TensorTrain()
    ```

    或者，從 `auto_mpo` 的建構過程中得到。

    ```python
    # H 是一個 polyOp 物件
    mpo_tensor_train = H.to_tensorTrain()
    ```

2.  **查詢元素值**: 使用 `eval` 方法。

    ```python
    # 查詢索引為 (1, 0, 1, ..., 1) 的張量元素值
    value = my_tensor_train.eval([1, 0, 1, ..., 1])
    print(f"張量元素值為: {value}")
    ```

3.  **計算總和**: 使用 `sum` 方法。

    ```python
    # 計算高維積分或配分函數 Z
    total_sum = my_tensor_train.sum()
    print(f"張量總和為: {total_sum}")
    ```

4.  **壓縮**: 使用壓縮方法來降低其複雜度。

    ```python
    print(f"壓縮前的最大鍵結維度: {my_tensor_train.max_bond_dimension}")
    # 使用 CI 壓縮到指定的容忍度
    my_tensor_train.compressCI(tolerance=1e-8)
    print(f"壓縮後的最大鍵結維度: {my_tensor_train.max_bond_dimension}")
    ```

### 概念驗證測試

要驗證 `TensorTrain` 物件的功能，可以建構一個簡單的、性質已知的張量鏈，然後對其進行操作並比對結果。

1.  **建構一個已知的張量鏈**: 手動建立一個代表簡單函數（例如 `F(σ₁, σ₂) = σ₁ + σ₂`）的 `TensorTrain`。這需要手動設定其 `cores` 列表中的小型矩陣。
2.  **驗證 `eval`**:
      * 手動計算 `F(1, 0)` 的理論值（應為 1）。
      * 呼叫 `my_tensor_train.eval([1, 0])`，並驗證其返回值是否與理論值相等。
3.  **驗證 `sum`**:
      * 手動計算該張量的所有元素總和。例如，如果 `σ` 的每個分量都是 0 或 1，則總和為 `F(0,0)+F(0,1)+F(1,0)+F(1,1)`。
      * 呼叫 `my_tensor_train.sum()`，並驗證其返回值是否與理論總和相等。
4.  **驗證壓縮**:
      * 建構一個可以被壓縮的張量鏈（例如，透過將兩個低秩張量鏈相加得到一個高秩但可壓縮的張量鏈）。
      * 記錄其壓縮前的 `bond_dimensions`。
      * 呼叫 `compressCI()` 或 `compressSVD()`。
      * 檢查壓縮後的 `bond_dimensions` 是否如預期般地減小了。
      * （可選）隨機選取幾個索引，比較壓縮前後 `eval` 的結果，確保誤差在 `tolerance` 範圍內。



## `tensor/tensor_ci.py`

此組件是 **TCI (Tensor Cross Interpolation) 演算法的主驅動程式**。它定義了 `TensorCI` 類別（可能有多個版本，如 `TensorCI1`），這個類別負責協調專案中的所有其他組件，執行迭代式的學習與優化過程，最終將一個高維的 `TensorFunction` 物件轉換為一個緊湊且高效的 `TensorTrain` 物件。

### 物件與功能論述

  * **`TensorCI` 物件**:
      * **核心用途**: 此物件是 TCI 演算法的**總指揮**或**協調器 (Orchestrator)**。它從一個初始的猜測開始（通常是一個秩為 1 的張量鏈），然後透過一系列的「掃描」(sweep) 過程，迭代地改進這個張量鏈，直到其能夠以指定的精度來近似原始的高維函數。

      * **論文連結**: `TensorCI` 物件的行為直接實現了論文第 4.3 節「2-site TCI algorithms」中所描述的核心演算法。整個演算法的流程被封裝在這個物件的方法中：

          * **初始化**: 對應演算法的第 (1) 步，從一個初始樞紐 (initial pivot) 開始，建立一個初步的、未經優化的張量鏈結構。
          * **迭代/掃描 (`iterate` 方法)**: 這正是論文演算法的第 (2) 步，即「來回掃描」(Sweeping back and forth)。每一次呼叫 `iterate` 方法，就相當於演算法沿著張量鏈從左到右或從右到左完成一次對所有「鍵結」(bond) 的更新。
          * **鍵結更新 (`_update_bond` 內部方法)**: 這實現了單次更新的詳細邏輯。它會選定一個鍵結，將相鄰的兩個核心張量合併，形成一個局部的 `Π_l` 張量視圖（透過 `MatrixInterface`），然後呼叫 `prrLU` 分解器 (`mat_decomp.py`) 來找到這個局部問題的最佳樞紐，最後用這個結果來更新整個張量鏈的結構。

      * **關鍵屬性**:

          * **`tensor_func`**: 儲存了要進行分解的目標 `TensorFunction` 物件。
          * **`index_set`**: 一個 `IndexSet` 物件，儲存並管理當前所有鍵結的樞紐索引。這是演算法的「狀態」，記錄了學習到的關於張量重要區域的資訊。
          * **`param`**: 一個參數物件，儲存了演算法的設定，如收斂容忍度 `tolerance`、最大鍵結維度 `max_bond_dimension` 等。
          * **`pivotError`**: 一個列表，記錄了每次 `iterate` 掃描後的近似誤差。這個屬性對於監控演算法的收斂過程至關重要。

      * **關鍵方法**:

          * **`__init__(...)`**: 建構函式。它接收 `TensorFunction` 和演算法參數，並進行初始化，建立一個初始的 `IndexSet` 和張量鏈結構。
          * **`iterate()`**: 演算法的核心驅動方法。**每呼叫一次，就會執行一次完整的掃描（例如，從左到右）**，更新所有的鍵結，並將新的近似誤差附加到 `pivotError` 列表中。
          * **`get_TensorTrain()`**: 在演算法收斂後，呼叫此方法可以獲取最終優化好的 `TensorTrain` 物件，這也是 TCI 演算法的最終產出。

### 如何呼叫與使用

使用者透過一個清晰的流程來驅動 TCI 演算法：

1.  **準備輸入**: 建立一個 `TensorFunction` 物件來包裝你的目標函數。

    ```python
    from comphy_project.interface.tensorfuc import TensorFunction

    def my_function(indices):
        # ... 你的高維函數邏輯 ...
        return result
        
    tensor_func = TensorFunction(my_function)
    ```

2.  **實例化 `TensorCI`**: 用 `TensorFunction` 和相關參數來建立 `TensorCI` 物件。

    ```python
    from comphy_project.tensor.tensor_ci import TensorCI

    # 設定參數，例如收斂容忍度
    params = {"tolerance": 1e-8, "max_bond_dimension": 50}

    tci_solver = TensorCI(tensor_func, shape_of_tensor, **params)
    ```

3.  **執行迭代迴圈**: 在一個迴圈中重複呼叫 `iterate()` 方法，直到滿足收斂條件（例如，誤差不再下降或達到最大掃描次數）。

    ```python
    MAX_SWEEPS = 20
    for i in range(MAX_SWEEPS):
        print(f"--- 執行第 {i+1} 次掃描 ---")
        tci_solver.iterate()
        current_error = tci_solver.pivotError[-1]
        print(f"當前誤差: {current_error}")
        
        # 簡單的收斂判斷
        if current_error < params["tolerance"]:
            print("已達到收斂容忍度！")
            break
    ```

4.  **獲取結果**: 從 `TensorCI` 物件中提取最終的 `TensorTrain`。

    ```python
    final_tensor_train = tci_solver.get_TensorTrain()
    ```

### 概念驗證測試

要驗證 `TensorCI` 的功能，重點在於確認其**收斂行為**和最終結果的**準確性**，這直接反映了其是否成功地「學習」到了張量的低秩結構。

1.  **選擇一個已知低秩的函數**: 建立一個理論上可以用低秩張量鏈精確表示的 `TensorFunction`，例如一個可分離函數 `F(σ₁, σ₂, σ₃) = f(σ₁) + g(σ₂) + h(σ₃)`。理論上，這種函數的張量鏈秩應該很小。
2.  **執行 TCI 流程**: 按照上述「如何呼叫」的步驟來執行完整的 TCI 分解。
3.  **監控收斂過程**: 在迭代迴圈中，打印出 `tci_solver.pivotError`。理論上，這個誤差值應該會**快速下降**，並在幾次掃描後收斂到一個接近機器精度的平台區。這證明了演算法的有效性。
4.  **檢查最終的秩**: 取得最終的 `final_tensor_train` 後，檢查其 `bond_dimensions`。對於一個低秩函數，這些維度應該都很小（例如，小於 5），這驗證了演算法的**秩揭示 (rank-revealing)** 能力，如論文所述。
5.  **驗證準確性**: 隨機挑選幾個索引 `sigma`，分別用原始的 `my_function` 和 `final_tensor_train.eval(sigma)` 來計算其值。比對這兩個值，它們的差異應該在設定的 `tolerance` 之內。這證明了 TCI 產生了一個準確的近似。
