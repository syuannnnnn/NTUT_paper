# Instruction
‼️交接光碟有放簡報連結+PDF檔，若需詳細流程請參考 140 裡: \digital_library_3\圖學實驗室_畢業交接光碟_112\交接光碟_112598048_曾詠暄\70_程式碼_資料 

‼️tsr 資料夾是 TripoSR github 裡面的 tsr，裡面是我修改過後的程式

‼️github 這邊僅放上有更動過或需要修改的程式碼

**步驟**

**1.** Clone TripoSR 官方 github

**2.** 架設環境、執行 TripoSR 文件中 example.py 是否可以正常執行 (詳細建置相關教學已放進 140 "70_程式碼_資料" 檔案夾)

**3.1. lora 微調:**

**訓練** 要改下面這幾個檔案: (新增一個訓練程式碼，執行LoRA微調訓練 ! )
- TripoSR/tsr/models/transformer/attention.py
- TripoSR/tsr/models/transformer/basic_transformer_block.py
- TripoSR/tsr/models/transformer/transformer_1d.py
  
(以上皆新增需 lora 相關參數，可以參考我的訓練程式碼中的註解)

- TripoSR/tsr/system.py : def forward() 形狀的地方 
- TripoSR/tsr/utils.py : def __call__()
- TripoSR/tsr/run.py : line 122 model 部分新增lora相關參數
- gradio_app.py
  
**評估** 於 run.py 的 parser.add_argument 新增:
- "--align" 用於對其點雲
- "--lora-weights" 用於使用由 lora 微調訓練後權重
- "--ground-truth" 用於計算指標

**3.2. Knowledge Distillation 微調** (是獨立的 不是延續 LoRA ! )

- 需要 Trellis output + TripoSR output + ground truth 一起算損失函數，所以要先有Trellis的輸出結果
- TripoSR 不用更動程式碼，直接拿他的原始輸出

**訓練** 一樣新增一個訓練程式碼:
- 主要是損失函數的配置
- 訓練集路徑設置要正確
- cfg.use_lora=false
  
**評估** 
- 和 lora 內容差不多，但要記得 cfg.use_lora=false


**Docker 執行**

建立 : docker build -t <image的名字> . 

執行 : docker run -it <image的名字>
