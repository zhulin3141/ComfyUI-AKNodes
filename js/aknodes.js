import { app } from "../../scripts/app.js";

app.registerExtension({
	name: "AKNodes.aknodes",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if(!nodeData?.category?.startsWith("AKNodes")) {
			return;
		}
		switch (nodeData.name) {
			
			case "EmptyLatentFromImageDimensions":
				const onGetImageSizeConnectInput = nodeType.prototype.onConnectInput;
				nodeType.prototype.onConnectInput = function (targetSlot, type, output, originNode, originSlot) {
					const v = onGetImageSizeConnectInput? onGetImageSizeConnectInput.apply(this, arguments): undefined

					this.outputs[1]["label"] = "width"
					this.outputs[2]["label"] = "height" 
					return v;
				}
				//const onGetImageSizeExecuted = nodeType.prototype.onExecuted;
				const onGetImageSizeExecuted = nodeType.prototype.onAfterExecuteNode;
				nodeType.prototype.onExecuted = function(message) {
					const r = onGetImageSizeExecuted? onGetImageSizeExecuted.apply(this,arguments): undefined
					let values = message["text"].toString().split('x').map(Number);

					this.outputs[1]["label"] = values[0] + " width"
					this.outputs[2]["label"] = values[1] + " height"
					return r
				}
				break;

		}		
		
	}
});