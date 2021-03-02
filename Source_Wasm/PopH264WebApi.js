

function GetYuv_8_8_8_PlaneDataMetas(Width,Height)
{
	const lw = Width;
	const lh = Height;
	const cw = lw / 2;
	const ch = lh / 2;
	const lsz = lw * lh;
	const csz = cw * ch;
	
	const Luma = {};
	Luma.Width = Width;
	Luma.Height = Height;
	Luma.Position = 0;
	Luma.Channels = 1;
	Luma.Format = 'Greyscale';
	Luma.Size = Luma.Width * Luma.Height * Luma.Channels;
	
	const ChromaU = {};
	ChromaU.Width = Width/2;
	ChromaU.Height = Height/2;
	ChromaU.Channels = 1;
	ChromaU.Position = Luma.Position + Luma.Size;
	ChromaU.Format = 'ChromaU';
	ChromaU.Size = ChromaU.Width * ChromaU.Height * ChromaU.Channels;

	const ChromaV = Object.assign({},ChromaU);
	ChromaV.Format = 'ChromaV';
	ChromaV.Position = ChromaU.Position + ChromaU.Size;
	
	return [Luma,ChromaU,ChromaV];
}

function GetYuv_8_8_8_BufferSize(Width,Height)
{
	const Meta = GetYuv_8_8_8_PlaneDataMetas(Width,Height);
	const LastPlaneMeta = Meta.pop();
	const End = LastPlaneMeta.Position + LastPlaneMeta.Size;
	return End;
}



export class Decoder
{
	constructor(DecoderSerial,Params={})
	{
		this.Params = Params;

		//this.player = new MP4Player(new Stream(src), useWorkers, webgl, render);
		const AvcParams = {};
		AvcParams.workerFile = "PopH264/Decoder.js";
		AvcParams.useWorker = true;
		
		//	avoid error for now
		//AvcParams.webgl = false;
		
		//AvcParams.reuseMemory = true;
		//AvcParams.size = {};
		//AvcParams.size.width = 640;
		//AvcParams.size.height = 368;
		//AvcParams.preserveDrawingBuffer = true;
	
		this.OnDecoderReadyPromise = Pop.CreatePromise();

		this.avc = new Player( AvcParams, this.OnDecoderReady.bind(this) );
		this.avc.onPictureDecoded = this.OnPictureDecoded.bind(this);
		
		this.DecodedImageQueue = new Pop.PromiseQueue('DecodedImageQueue');
		
		//	data needs to be sent in this order, otherwise decoder will get stuck
		this.SentSps = false;
		this.SentPps = false;
		this.SentIdr = false;
	}

	OnDecoderReady()
	{
		this.OnDecoderReadyPromise.Resolve();
	}
	
	async WaitForDecoderReady()
	{
		return this.OnDecoderReadyPromise;
	}
	
	OnPictureDecoded(PixelBytes,Width,Height,PacketMeta)
	{
		const Frame = {};
		Frame.Planes = [];
		Frame.FrameNumber = null;
		
		if ( PacketMeta && PacketMeta.FrameNumber !== undefined )
		{
			Frame.FrameNumber = PacketMeta.FrameNumber;
		}
		else
		{
			Pop.Warning(`Decoded frame missing frame number; Meta=${JSON.stringify(PacketMeta)}`);
		}
		
		const YuvSize = GetYuv_8_8_8_BufferSize(Width,Height);

		if ( Width * Height * 4 == PixelBytes.length )
		{
			const Format = 'RGBA';
			const FrameImage = this.AllocImage( Width, Height, Format, PixelBytes );
			Frame.Planes.push(FrameImage);
		}
		else if ( PixelBytes.length == YuvSize )
		{
			//	split data into planes
			const Planes = this.SplitPixelBytesIntoYuv_8_8_8(Width,Height,PixelBytes);
			Frame.Planes.push(...Planes);
		}
		else
		{
			//	raw data as greyscale
			const UnrolledHeight = PixelBytes.length/Width;
			const Format = 'Greyscale';
			const FrameImage = this.AllocImage( Width, UnrolledHeight, Format, PixelBytes );
			Frame.Planes.push(FrameImage);
		}
				
		this.DecodedImageQueue.Push(Frame);
	}

	PeekNextFrame()
	{
		return this.DecodedImageQueue.PendingValues[0];
	}

	async WaitForNextFrame(LatestOnly=false)
	{
		if (LatestOnly)
			return this.DecodedImageQueue.WaitForLatest();
		else
			return this.DecodedImageQueue.WaitForNext();
	}
	
	AllocImage(Width,Height,Format,PixelBytes)
	{
		const Image = new Pop.Image(`H264 Decoder image`);
		Image.WritePixels( Width, Height, PixelBytes, Format );
		return Image;
	}
	
	FreeImage(Image)
	{
		//	nothing to do atm
	}
		
	//	todo: make this part of Pop.Image.SplitPlanes()
	SplitPixelBytesIntoYuv_8_8_8(Width,Height,PixelBuffer)
	{
		function MetaToImage(PlaneMeta)
		{
			const PlanePixels = PixelBuffer.slice( PlaneMeta.Position, PlaneMeta.Position + PlaneMeta.Size );
			const Width = PlaneMeta.Width;
			const Height = PlaneMeta.Height;
			const Format = PlaneMeta.Format;
			const Image = this.AllocImage( Width, Height, Format, PlanePixels );
			return Image;
		}
		
		const Meta = GetYuv_8_8_8_PlaneDataMetas(Width,Height);
		const Planes = Meta.map( MetaToImage.bind(this) );
		return Planes;
	}

	ReleaseFrame(Frame)
	{
		//	free images back to pool
		for ( let Image of Frame.Planes )
			this.FreeImage( Image );
		Frame.Planes = [];
	}
	
	PushNalu(Nalu,FrameMeta)
	{
		const Meta = Pop.H264.GetNaluMeta(Nalu);
		let PushData = true;
		
		const IsSps = ( Meta.Content == Pop.H264.SPS);
		const IsPps = ( Meta.Content == Pop.H264.PPS); 
		const IsIdr = ( Meta.Content == Pop.H264.Slice_CodedIDRPicture); 

		if ( !this.SentSps )
		{
			if ( !IsSps )
			{
				Pop.Debug(`Skipping ${Meta.Content}: no SPS yet`);
				return;
			}
		}
		else if ( !this.SentPps )
		{
			if ( !IsPps )
			{
				Pop.Debug(`Skipping ${Meta.Content}: no PPS yet`);
				return;
			}
		}
		else if ( !this.SentIdr )
		{
			if ( !IsIdr )
			{
				Pop.Debug(`Skipping ${Meta.Content}: no IDR yet`);
				return;
			}
		}
		
		this.avc.decode(Nalu,FrameMeta);

		this.SentSps = this.SentSps || IsSps;
		this.SentPps = this.SentPps || IsPps;
		this.SentIdr = this.SentIdr || IsIdr;
	}
	
	//	unity interface
	PushFrameData(H264Packet,FrameNumber)
	{
		const Meta = {};
		Meta.FrameNumber = FrameNumber;
		
		//	data needs to be sent in this order, otherwise decoder will get stuck
		const Nalus = Pop.H264.SplitNalus(H264Packet);
		
		Nalus.forEach( Nalu => this.PushNalu(Nalu,Meta) );
	}
	
	Decode(H264Packet,FrameNumber=null)
	{
		throw `Decode() is deprecated, now use PushFrameData and provide frame time/number`;
	}
	/*
	constructor()
	{
		this.Init().catch(Pop.Warning);
		
	}
	
	async Init()
	{
		const WasmCode = await Pop.LoadFileAsArrayBufferAsync('PopH264/TinyH264.wasm.asset');
		this.WasmModule = await LoadWasmModule(WasmCode);
		
		this.pStorage = this.WasmModule._h264bsdAlloc()
		this.pWidth = this.WasmModule._malloc(4)
		this.pHeight = this.WasmModule._malloc(4)
		this.pPicture = this.WasmModule._malloc(4)
		this._decBuffer = this.tinyH264Module._malloc(1024 * 1024)
		this.WasmModule._h264bsdInit(this.pStorage, 0)
	}
	
	Decode(H264Bytes)
	{
		const WasmH264Bytes = this.WasmModule.HeapAllocArray(Uint8Array,H264Bytes.length);
		WasmH264Bytes.set(H264Bytes,0,H264Bytes.length);
		
		//	this can throw first time if HeapAlloc resizes
		WasmModule.Instance.exports.Decode(WasmH264Bytes.byteOffset,Yuv8_8_8.byteOffset,w,h,DepthMin,DepthMax);
	}
	*/
}
