
# AGENTS.md

## 目的

このリポジトリでは、AI/ML初心者が Codex に自然言語で依頼するだけで、
**ブラウザ上で動くAIモデル機能を、WebGPUを使って実装できること**を最優先にする。

対象は主に以下。

- Depth Anything V2: 深度推定
- SAM / SlimSAM: 画像セグメンテーション
- DINOv2: 画像特徴量・画像類似度
- Whisper: 音声文字起こし
- CLAP: 音声とテキストの意味類似度
- YOLOv10: 物体検出・個数カウント

UIデザインはユーザーが決める。
Codexは、見た目の提案・装飾・レイアウト設計を勝手に追加しない。

---

## 最重要方針

ユーザーが「○○を実装したい」と言ったら、
説明だけで終わらず、**ブラウザで実際に動くコードまで作ること**。

原則として以下を守る。

1. 推論はブラウザ内で完結させる。
2. WebGPUを使用する。
3. サーバー推論やPythonバックエンドを勝手に追加しない。
4. フレームワーク指定がなければ、まずは HTML + JavaScript で実装する。
5. ビルド環境が不要な構成を優先する。
6. UIは最小限のDOM要素だけ作り、デザインには踏み込まない。
7. WebGPU未対応時はCPUへ黙ってフォールバックしない。明示的にエラーにする。
8. モデルID・前処理・後処理は、下記の既知の動作パターンを優先する。
9. 初回モデル取得にはネット接続が必要であることを前提にする。
10. カメラ・マイクを使う機能は HTTPS 配信を前提にする。

---

## 基本スタック

特に指定がなければ、以下を基準にする。

```html
<script type="module">
import {
  pipeline,
  env,
  RawImage,
  Tensor,
  SamModel,
  AutoProcessor,
  AutoModel,
  AutoTokenizer,
  ClapTextModelWithProjection,
  ClapAudioModelWithProjection,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

env.allowLocalModels = false;
</script>
```

原則:

```js
device: "webgpu"
```

モデルは必要になるまでロードしない。
一度ロードしたモデルは同じ処理中では再利用する。

---

# 依頼内容から使うモデルを決める

以下の意図を読み取ってモデルを選ぶ。

| ユーザーの意図・語句 | 使用モデル |
|---|---|
| 深度、奥行き、depth、距離感、深度マップ | Depth Anything V2 |
| セグメント、領域選択、物体選択、切り抜き、マスク | SAM / SlimSAM |
| 画像類似度、似ている画像、画像embedding、特徴量 | DINOv2 |
| 文字起こし、音声認識、speech to text | Whisper |
| 音と文章の類似度、音の意味判定、音響embedding | CLAP |
| 物体検出、bbox、何が写っているか、個数を数える | YOLOv10 |

複数の候補がある場合でも、ユーザーの目的に最も直接対応する1モデルから実装する。

---

# 共通: WebGPU対応

実装時には最初に WebGPU 利用可否を確認する。

```js
async function requireWebGPU() {
  if (!navigator.gpu) {
    throw new Error("WebGPU が利用できません");
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error("WebGPU adapter を取得できません");
  }

  return adapter;
}
```

CPU推論へ自動フォールバックしない。

必要なら以下も確認してよい。

```js
const adapter = await requireWebGPU();
console.log(
  "maxBufferSize:",
  Math.round(adapter.limits.maxBufferSize / 1024 ** 2),
  "MiB"
);
```

---

# 共通: ブラウザで安定させるためのルール

## 1. 同時実行を防ぐ

モデルロード中や推論中に、別の推論を重ねて開始しない。

```js
let busy = false;

async function runExclusive(fn) {
  if (busy) return;
  busy = true;
  try {
    await fn();
  } finally {
    busy = false;
  }
}
```

---

## 2. 大きい画像をそのまま入れない

スマートフォン写真を原寸のままモデルへ渡さない。

用途別の目安:

- Depth Anything: 長辺 1024px 程度
- SAM: 長辺 960px 程度
- YOLO: 長辺 960px 程度
- DINO: 長辺 700px 程度でも十分

```js
async function fileToCanvas(file, maxSide = 960) {
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.src = url;
  await img.decode();
  URL.revokeObjectURL(url);

  const scale = Math.min(
    1,
    maxSide / Math.max(img.naturalWidth, img.naturalHeight)
  );

  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(img.naturalWidth * scale));
  canvas.height = Math.max(1, Math.round(img.naturalHeight * scale));

  canvas
    .getContext("2d")
    .drawImage(img, 0, 0, canvas.width, canvas.height);

  return canvas;
}
```

---

## 3. GPU Tensorを明示的に解放する

推論結果や一時Tensorを放置しない。

```js
tensor.dispose?.();
```

ネストした結果をまとめて破棄する必要がある場合:

```js
function disposeTensorTree(value, seen = new Set()) {
  if (!value || typeof value !== "object" || seen.has(value)) return;
  seen.add(value);

  if (typeof value.dispose === "function") {
    value.dispose();
    return;
  }

  if (ArrayBuffer.isView(value) || value instanceof ArrayBuffer) return;

  if (Array.isArray(value)) {
    for (const x of value) disposeTensorTree(x, seen);
    return;
  }

  for (const x of Object.values(value)) {
    disposeTensorTree(x, seen);
  }
}
```

---

## 4. Safariではモデル切替時のGPUメモリ残留を警戒する

複数の重いWebGPUモデルを1ページで切り替える構成では、
`dispose()` 後も Safari / WebGPU / ONNX Runtime 側にGPUメモリがしばらく残る場合がある。

そのため、複数モデルを切り替えるアプリでは、
**モデル間の切替時にページを再読み込みしてJS realmごと破棄する方式を優先する。**

例:

```js
function switchModelPage(tab) {
  location.hash = tab;
  location.reload();
}
```

単一モデルしか使わないページなら、この処理は不要。

---

## 5. ページ離脱時にリソースを解放する

```js
window.addEventListener("pagehide", () => {
  model?.dispose?.();

  for (const url of objectUrls) {
    URL.revokeObjectURL(url);
  }

  if (cameraStream) {
    for (const track of cameraStream.getTracks()) {
      track.stop();
    }
  }
});
```

---

## 6. Object URLを放置しない

```js
const url = URL.createObjectURL(blob);

// 使用終了後
URL.revokeObjectURL(url);
```

画像・音声プレビューを差し替える場合は、前のURLを先に破棄する。

---

## 7. モデルのダウンロード進捗を受け取る

長いロード時間を無反応にしない。

```js
function progressCallback(x) {
  if (x.status === "progress" && Number.isFinite(x.progress)) {
    console.log(`${x.progress.toFixed(0)}%`, x.file ?? "");
  }
}
```

モデルロード時:

```js
progress_callback: progressCallback
```

UIデザインは不要だが、進捗を表示できる状態にはしておく。

---

# SAM / SlimSAM

## このモデルを使う依頼

以下のような依頼なら SAM を使う。

- 「画像をクリックして物体を選択したい」
- 「セグメントしたい」
- 「領域を選択したい」
- 「物体を切り抜きたい」
- 「クリックした場所のマスクが欲しい」
- 「前景点と背景点を指定して領域を絞りたい」

## 使用モデル

```js
const SAM_MODEL_ID = "Xenova/slimsam-77-uniform";
```

```js
device: "webgpu",
dtype: "fp16"
```

## 重要

SAMは毎クリック時に画像全体を再エンコードしない。

処理を2段階に分ける。

1. 画像を1回だけエンコードして image embeddings を保存
2. クリック点が変わるたびに mask decoder だけ実行

これを守ること。

## ロード

```js
const samModel = await SamModel.from_pretrained(
  "Xenova/slimsam-77-uniform",
  {
    device: "webgpu",
    dtype: "fp16",
    progress_callback: progressCallback,
  }
);

const samProcessor = await AutoProcessor.from_pretrained(
  "Xenova/slimsam-77-uniform"
);
```

## 画像エンコード

```js
const raw = RawImage.fromCanvas(inputCanvas);
const processed = await samProcessor(raw);
const embeddings = await samModel.get_image_embeddings(processed);
```

以下はクリック中ずっと保持してよい。

```js
processed
embeddings
```

画像を差し替えるときは古いものを dispose する。

## クリック座標

画面表示サイズとCanvas内部解像度を混同しない。

pointer eventから0〜1の正規化座標を作る。

```js
const rect = canvas.getBoundingClientRect();

const x = Math.max(
  0,
  Math.min(1, (event.clientX - rect.left) / rect.width)
);

const y = Math.max(
  0,
  Math.min(1, (event.clientY - rect.top) / rect.height)
);
```

点は以下の形で保持する。

```js
points.push({
  x,
  y,
  label: 1,
});
```

- `label: 1` = 対象物側の点
- `label: 0` = 除外したい背景側の点

## SAM入力座標へ変換

```js
const reshaped = processed.reshaped_input_sizes[0];

const pointValues = points.flatMap((p) => [
  p.x * reshaped[1],
  p.y * reshaped[0],
]);

const labels = points.map((p) => BigInt(p.label));
```

Tensor:

```js
const inputPoints = new Tensor(
  "float32",
  pointValues,
  [1, 1, points.length, 2]
);

const inputLabels = new Tensor(
  "int64",
  labels,
  [1, 1, points.length]
);
```

## mask decoder

```js
const outputs = await samModel({
  ...embeddings,
  input_points: inputPoints,
  input_labels: inputLabels,
});
```

## 元画像サイズに戻す

```js
const masks = await samProcessor.post_process_masks(
  outputs.pred_masks,
  processed.original_sizes,
  processed.reshaped_input_sizes
);
```

Transformers.js で扱いやすい形へ変換:

```js
const maskImage = RawImage.fromTensor(masks[0][0]);
const scores = Array.from(outputs.iou_scores.data);
```

SAMは通常、複数のマスク候補を返す。

自動選択するなら最大scoreを使う。

```js
let best = 0;

for (let i = 1; i < scores.length; i++) {
  if (scores[i] > scores[best]) {
    best = i;
  }
}
```

必要なら候補をすべてユーザーへ返せる構造にしておく。

## SAMの後始末

毎回生成する一時Tensorはすぐ破棄する。

```js
inputPoints.dispose?.();
inputLabels.dispose?.();
disposeTensorTree(outputs);
disposeTensorTree(masks);
```

画像を変更するとき:

```js
disposeTensorTree(embeddings);
disposeTensorTree(processed);
```

---

# Depth Anything V2

## このモデルを使う依頼

- 「深度マップを作りたい」
- 「画像の奥行きを推定したい」
- 「depth estimationしたい」
- 「手前と奥を可視化したい」

## 使用モデル

```js
const DEPTH_MODEL_ID =
  "onnx-community/depth-anything-v2-small";
```

推奨:

```js
device: "webgpu",
dtype: "q4"
```

## ロード

```js
const depthPipe = await pipeline(
  "depth-estimation",
  "onnx-community/depth-anything-v2-small",
  {
    device: "webgpu",
    dtype: "q4",
    progress_callback: progressCallback,
  }
);
```

## 推論

```js
const output = await depthPipe(inputCanvas);
```

主に使う出力:

```js
output.depth
```

`output.depth` は表示用の画像として扱える。

不要になったTensor:

```js
output.predicted_depth?.dispose?.();
```

画像は長辺1024px程度へ縮小してから渡すことを推奨する。

---

# DINOv2

## このモデルを使う依頼

- 「2枚の画像が似ているか知りたい」
- 「画像embeddingを取得したい」
- 「画像特徴量を比較したい」
- 「基準画像にどれくらい似ているか数値化したい」

## 使用モデル

```js
const DINO_MODEL_ID = "onnx-community/dinov2-small";
```

推奨:

```js
device: "webgpu",
dtype: "q4"
```

## ロード

```js
const dinoPipe = await pipeline(
  "image-feature-extraction",
  "onnx-community/dinov2-small",
  {
    device: "webgpu",
    dtype: "q4",
    progress_callback: progressCallback,
  }
);
```

## DINOv2で画像全体の特徴量を取る

DINOv2の出力をそのままflattenして比較しない。

出力が `[1, tokens, hidden]` の場合、
先頭の CLS token を画像全体のベクトルとして使う。

```js
function dinoClsVector(tensor) {
  if (tensor.dims.length !== 3 || tensor.dims[0] !== 1) {
    throw new Error(
      `Unexpected DINO output shape: [${tensor.dims.join(", ")}]`
    );
  }

  const hidden = tensor.dims[2];
  return tensor.data.subarray(0, hidden);
}
```

## cosine similarity

```js
function cosine(a, b) {
  if (a.length !== b.length) {
    throw new Error(
      `embedding size mismatch: ${a.length} vs ${b.length}`
    );
  }

  let dot = 0;
  let aa = 0;
  let bb = 0;

  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    aa += a[i] * a[i];
    bb += b[i] * b[i];
  }

  return dot / Math.sqrt(aa * bb);
}
```

利用例:

```js
const a = await dinoPipe(referenceCanvas);
const b = await dinoPipe(inputCanvas);

const score = cosine(
  dinoClsVector(a),
  dinoClsVector(b)
);

a.dispose?.();
b.dispose?.();
```

cosine similarityは確率ではない。
「85%の確率」などと表示しない。

---

# Whisper

## このモデルを使う依頼

- 「音声を文字起こししたい」
- 「ブラウザでspeech to textしたい」
- 「録音してテキスト化したい」

## 使用モデル

```js
const WHISPER_MODEL_ID =
  "onnx-community/whisper-tiny";
```

推奨:

```js
device: "webgpu",
dtype: "q8"
```

## ロード

```js
const whisper = await pipeline(
  "automatic-speech-recognition",
  "onnx-community/whisper-tiny",
  {
    device: "webgpu",
    dtype: "q8",
    progress_callback: progressCallback,
  }
);
```

## 音声前処理

ブラウザで読み込んだ音声を以下に変換する。

- mono
- 16 kHz
- 長すぎる音声は切る
- 初心者向けサンプルでは最大30秒程度を推奨

音声チャンネルが複数ある場合は平均する。

```js
const mono = new Float32Array(frames);

for (let ch = 0; ch < decoded.numberOfChannels; ch++) {
  const data = decoded.getChannelData(ch);

  for (let i = 0; i < frames; i++) {
    mono[i] += data[i] / decoded.numberOfChannels;
  }
}
```

サンプルレートが16kHzでなければリサンプリングする。

線形補間で十分。

## 推論

```js
const result = await whisper(audioFloat32Array, {
  task: "transcribe",
});
```

結果をユーザーへ返す。

---

# CLAP

## このモデルを使う依頼

- 「この音が○○っぽいか判定したい」
- 「音と文章の意味類似度を出したい」
- 「音声embeddingとtext embeddingを比較したい」
- 「挨拶音っぽいか、拍手っぽいか等を比較したい」

## 使用モデル

```js
const CLAP_MODEL_ID =
  "Xenova/clap-htsat-unfused";
```

推奨:

```js
device: "webgpu",
dtype: "q8"
```

## 重要

CLAPでは、

1. text encoderで比較対象の文章をembedding化
2. audio encoderで音声をembedding化
3. cosine similarityを計算

という流れにする。

## text embedding

```js
const tokenizer = await AutoTokenizer.from_pretrained(
  "Xenova/clap-htsat-unfused"
);

const textModel =
  await ClapTextModelWithProjection.from_pretrained(
    "Xenova/clap-htsat-unfused",
    {
      device: "webgpu",
      dtype: "q8",
      progress_callback: progressCallback,
    }
  );

const inputs = tokenizer(
  ["a person greeting or saying hello"],
  {
    padding: true,
    truncation: true,
  }
);

const { text_embeds } = await textModel(inputs);
const textVector = Float32Array.from(text_embeds.data);

text_embeds.dispose?.();
await textModel.dispose();
```

固定テキストとの比較なら、
text embeddingを一度作った後にtext modelを解放してよい。

これはGPUメモリ節約に有効。

CLAPは英語テキスト中心のため、
日本語の短い概念ラベルは、必要に応じて自然な英語説明文へ変換してからembeddingする。

## audio embedding

```js
const processor = await AutoProcessor.from_pretrained(
  "Xenova/clap-htsat-unfused"
);

const audioModel =
  await ClapAudioModelWithProjection.from_pretrained(
    "Xenova/clap-htsat-unfused",
    {
      device: "webgpu",
      dtype: "q8",
      progress_callback: progressCallback,
    }
  );
```

音声は原則:

- mono
- 48 kHz
- 初心者向けデモでは最大10秒程度

```js
const audioInputs = await processor(audioFloat32Array);

const { audio_embeds } =
  await audioModel(audioInputs);

const score = cosine(
  textVector,
  audio_embeds.data
);
```

後始末:

```js
audio_embeds.dispose?.();
disposeTensorTree(audioInputs);
```

CLAPのcosine similarityは確率ではない。

---

# YOLOv10

## このモデルを使う依頼

- 「画像の中の物体を検出したい」
- 「人やスマホの数を数えたい」
- 「bboxを出したい」
- 「何が何個あるか知りたい」

## 使用モデル

```js
const YOLO_MODEL_ID =
  "onnx-community/yolov10n";
```

推奨:

```js
device: "webgpu",
dtype: "q8"
```

## ロード

```js
const yoloModel = await AutoModel.from_pretrained(
  "onnx-community/yolov10n",
  {
    device: "webgpu",
    dtype: "q8",
    progress_callback: progressCallback,
  }
);

const yoloProcessor =
  await AutoProcessor.from_pretrained(
    "onnx-community/yolov10n"
  );
```

## 前処理

```js
const raw = RawImage.fromCanvas(inputCanvas);
const processed = await yoloProcessor(raw);
```

## 推論

```js
const { output0 } = await yoloModel({
  images: processed.pixel_values,
});
```

このモデルでは各予測を以下として扱う。

```text
[xmin, ymin, xmax, ymax, score, classId]
```

## bboxを元画像へ戻す

processor後のサイズと入力画像サイズは異なる。

```js
const [newH, newW] =
  processed.reshaped_input_sizes[0];

const xs = inputCanvas.width / newW;
const ys = inputCanvas.height / newH;
```

各bbox:

```js
const x = xmin * xs;
const y = ymin * ys;
const w = (xmax - xmin) * xs;
const h = (ymax - ymin) * ys;
```

ラベル:

```js
const label =
  yoloModel.config.id2label[classId];
```

confidence thresholdは用途に応じて変更できるようにする。

初心者向けの初期値としては `0.4` 程度でよい。

## 個数を数える

```js
const counts = new Map();

counts.set(
  label,
  (counts.get(label) ?? 0) + 1
);
```

## 後始末

```js
output0.dispose?.();
processed.pixel_values.dispose?.();
```

注意:
YOLOv10nの配布元モデルのライセンス条件は、
公開・配布する用途では必ず確認する。

---

# カメラ入力を実装する場合

画像系モデルで「今撮影」を使いたい場合は `getUserMedia()` を使う。

HTTPS前提。

```js
const stream =
  await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: {
        ideal: "environment",
      },
    },
    audio: false,
  });
```

スマートフォン向けでは背面カメラを優先する。

`video` には以下を付ける。

```html
<video autoplay muted playsinline></video>
```

撮影後はCanvasへ描画し、必ずstreamを停止する。

```js
for (const track of stream.getTracks()) {
  track.stop();
}
```

撮影画像もモデル投入前に長辺を制限する。

---

# マイク録音を実装する場合

```js
const stream =
  await navigator.mediaDevices.getUserMedia({
    audio: true,
  });

const recorder =
  new MediaRecorder(stream);
```

録音停止後:

```js
for (const track of stream.getTracks()) {
  track.stop();
}
```

録音データを `AudioContext.decodeAudioData()` で読み、
各モデルの必要サンプルレートへ変換する。

- Whisper: 16 kHz
- CLAP: 48 kHz

---

# モデルロードの共通形

pipelineで扱えるモデルでは、以下のようなキャッシュ関数を使ってよい。

```js
const modelCache = Object.create(null);

async function ensurePipeline(
  key,
  task,
  modelId,
  dtype
) {
  if (modelCache[key]) {
    return modelCache[key];
  }

  modelCache[key] = await pipeline(
    task,
    modelId,
    {
      device: "webgpu",
      dtype,
      progress_callback: progressCallback,
    }
  );

  return modelCache[key];
}
```

SAM、CLAP、YOLOのように専用classを使うモデルは、
pipelineへ無理に統一しない。

---

# 複数モデルを1つのHTMLに入れる場合

初心者向けで複数モデルを同一ページへまとめる場合は、
GPUメモリを最重要視する。

推奨:

- 同時に複数の重いモデルをロードしない
- 選択中モデルだけロードする
- モデル切替時はページreloadを許容する
- URL hash等で開くモデルを保持する

例:

```js
location.hash = "sam";
location.reload();
```

ページ再読み込み後にhashを見て、
そのモデルだけ初期化する。

モデルファイル自体はブラウザキャッシュに残るため、
通常は毎回フルダウンロードにはならない。

---

# エラー処理方針

初心者向けでも、不具合を隠さない。

以下は明示的に失敗させる。

- WebGPUがない
- adapterが取れない
- 入力画像がない
- 入力音声がない
- SAMで画像embeddingを作る前にクリックされた
- DINOの出力shapeが想定外
- embedding次元数が一致しない
- カメラ・マイクAPIが利用できない

予期しない例外を握りつぶさない。

---

# UIについて

UIはユーザーが決める。

Codexは以下をしない。

- CSSデザインの提案
- 配色の決定
- レイアウトの作り込み
- カードUIの追加
- アイコンライブラリの追加
- アニメーション追加
- Tailwind等の導入
- React/Vue等への勝手な変更

ただしAI機能を動かすために必要な最小DOMは追加してよい。

例:

- `<input type="file">`
- `<canvas>`
- `<audio>`
- `<video>`
- 実行ボタン
- 状態表示用の要素

既存UIがある場合は、そのUIへ処理を接続する。

---

# Codexが実装するときの進め方

依頼を受けたら以下の順に進める。

1. ユーザーの目的からモデルを選ぶ
2. 入力形式を確認する
3. WebGPUチェックを入れる
4. Transformers.js importを追加する
5. モデルを遅延ロードする
6. 必要な前処理を実装する
7. WebGPUで推論する
8. 必要な後処理を実装する
9. Tensor / stream / Object URLを解放する
10. ブラウザだけで実行できることを確認する

UIデザインはこの工程に含めない。

---

# 依頼例 → 実装方針

## 例1

ユーザー:

> SAMで画像の物体をクリックして選択したい

実装:

- SlimSAM
- `SamModel`
- `AutoProcessor`
- WebGPU / fp16
- 最初に画像embeddingを1回だけ計算
- click pointは0〜1で保持
- `reshaped_input_sizes` でSAM座標へ変換
- positive / negative points対応
- mask decoderだけ再実行
- `post_process_masks()` で元画像サイズへ戻す
- 複数候補はIoU score最大を既定にする
- Tensorを毎回disposeする

## 例2

ユーザー:

> 写真から深度マップを出したい

実装:

- Depth Anything V2 Small
- `pipeline("depth-estimation")`
- WebGPU / q4
- 入力画像長辺を1024px程度へ縮小
- `output.depth` を利用
- `predicted_depth` をdispose

## 例3

ユーザー:

> 2枚の写真がどれくらい似ているか知りたい

実装:

- DINOv2 Small
- `pipeline("image-feature-extraction")`
- WebGPU / q4
- CLS tokenを使用
- cosine similarityを計算
- similarityを確率として扱わない

## 例4

ユーザー:

> ブラウザで録音して文字起こししたい

実装:

- Whisper Tiny
- `pipeline("automatic-speech-recognition")`
- WebGPU / q8
- MediaRecorder
- mono / 16kHz
- 最大30秒程度
- `{ task: "transcribe" }`

## 例5

ユーザー:

> 音声が「挨拶」に近いか判定したい

実装:

- CLAP
- text encoderで比較文のembeddingを作る
- 固定文ならtext modelはembedding作成後に解放
- audio encoderをロード
- 音声をmono / 48kHzへ変換
- cosine similarityで比較
- 類似度を確率と呼ばない

## 例6

ユーザー:

> 写真に写っているスマホやPCの数を数えたい

実装:

- YOLOv10n
- AutoModel + AutoProcessor
- WebGPU / q8
- confidence threshold
- bboxを元画像座標へスケール
- `config.id2label` でラベル変換
- classごとにcount
- output tensorをdispose

---

# 完了条件

AI機能の実装は、少なくとも以下を満たして完了とする。

- ブラウザのみで実行可能
- WebGPUでモデルをロードしている
- 初回ロード後に推論できる
- 入力データの前処理が正しい
- 出力の後処理が正しい
- 同じモデルを毎回ロードし直さない
- 不要なTensorを解放している
- カメラ使用後にtrackを停止している
- マイク使用後にtrackを停止している
- Object URLを必要以上に残さない
- 画像サイズを適切に制限している
- 複数モデル構成ではGPUメモリ残留を考慮している
- UIデザインへ勝手に踏み込んでいない

---

# 迷ったときの原則

「綺麗なコード」より先に、
**初心者のブラウザで確実に動く小さい実装**を作る。

「高機能」より、
**1モデル・1目的・最小構成**を優先する。

WebGPU・前処理・後処理・メモリ解放は省略しない。
UIは省略してよい。