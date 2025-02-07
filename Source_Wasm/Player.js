/*


usage:

p = new Player({
  useWorker: <bool>,
  workerFile: <defaults to "Decoder.js"> // give path to Decoder.js
  webgl: true | false | "auto" // defaults to "auto"
});

// canvas property represents the canvas node
// put it somewhere in the dom
p.canvas;

p.webgl; // contains the used rendering mode. if you pass auto to webgl you can see what auto detection resulted in

p.decode(<binary>);


*/



// universal module definition
(function (root, factory) {
    if (typeof define === 'function' && define.amd) {
        // AMD. Register as an anonymous module.
        define(["./Decoder", "./YUVCanvas"], factory);
    } else if (typeof exports === 'object') {
        // Node. Does not work with strict CommonJS, but
        // only CommonJS-like environments that support module.exports,
        // like Node.
        module.exports = factory(require("./Decoder"), require("./YUVCanvas"));
    } else {
        // Browser globals (root is window)
        root.Player = factory(root.Decoder, root.YUVCanvas);
    }
}(this, function (Decoder)
{
	"use strict";


	var nowValue = Decoder.nowValue;

	//	gr: can I just make this a class?
	var Player = function(parOptions,OnDecoderReady)
	{
		var self = this;
		
		//	callback in some cases was being called before we had time to set the callback, so is now passed in as param
		this.OnDecoderReady = OnDecoderReady || function()	{	console.log(`OnDecoderReady.`);	};
		
		this._config = parOptions || {};

		this.nowValue = nowValue;

		this._config.workerFile = this._config.workerFile || "Decoder.js";


		var lastWidth;
		var lastHeight;
		var onPictureDecoded = function(buffer, width, height, infos) 
		{
			//	call overload
			self.onPictureDecoded(buffer, width, height, infos);
		};


		// provide size
		if (!this._config.size)
		{
			this._config.size = {};
		};
		this._config.size.width = this._config.size.width || 200;
		this._config.size.height = this._config.size.height || 200;

		if (this._config.useWorker)
		{
			var worker = new Worker(this._config.workerFile);
			this.worker = worker;
			function OnWorkerMessage(e) 
			{
				var data = e.data;
				if (data.consoleLog)
				{
					console.log(data.consoleLog);
					return;
				};
				if ( data.onDecoderReady )
				{
					self.OnDecoderReady();
					return;
				}
				onPictureDecoded.call(self, new Uint8Array(data.buf, 0, data.length), data.width, data.height, data.infos);
			}
			worker.addEventListener('message', OnWorkerMessage, false );

			const Options = 
			{
				rgb: false,
				memsize: this.memsize,
				reuseMemory: this._config.reuseMemory ? true : false,
			}
			worker.postMessage({type: "Player.js - Worker init", options: Options });

			//	gr: default off
			if (this._config.transferMemory)
			{
				this.decode = function(parData, parInfo)
				{
					// no copy
					// instead we are transfering the ownership of the buffer
					// dangerous!!!
					worker.postMessage({buf: parData.buffer, offset: parData.byteOffset, length: parData.length, info: parInfo}, [parData.buffer]); // Send data to our worker.
				};
			}
			else
			{
				this.decode = function(parData, parInfo)
				{
					// Copy the sample so that we only do a structured clone of the
					// region of interest
					var copyU8 = new Uint8Array(parData.length);
					copyU8.set( parData, 0, parData.length );
					worker.postMessage({buf: copyU8.buffer, offset: 0, length: parData.length, info: parInfo}, [copyU8.buffer]); // Send data to our worker.
				};
			};

			if (this._config.reuseMemory)
			{
				this.recycleMemory = function(parArray)
				{
					//this.beforeRecycle();
					worker.postMessage({reuse: parArray.buffer}, [parArray.buffer]); // Send data to our worker.
					//this.afterRecycle();
				};
			}
		}
		else
		{
			const Options = 
			{
				rgb:false
			};
			this.decoder = new Decoder(Options);

			//	gr: shouldn't this be calling with self/this?
			this.decoder.onPictureDecoded = onPictureDecoded;
			this.decoder.onDecoderReady = self.OnDecoderReady.bind(self);

			this.decode = function(parData, parInfo)
			{
				self.decoder.decode(parData, parInfo);
			};

		};


		lastWidth = this._config.size.width;
		lastHeight = this._config.size.height;
	};

	Player.prototype =
	{
		onPictureDecoded: function(buffer, width, height, Meta)
		{
			console.log(`Player.onPictureDecoded - Not overloaded`);
		},
	
		// call when memory of decoded frames is not used anymore
		recycleMemory: function(buf)
		{
		},
		/*beforeRecycle: function(){},
		afterRecycle: function(){},*/
	};

	return Player;

}));

