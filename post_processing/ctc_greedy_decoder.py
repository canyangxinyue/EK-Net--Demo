

class CTCGreedyDecoder():
    
    def __init__(self, ctc_blank=0):
        self.ctc_blank=ctc_blank
        
    def __call__(self, batch, output, label2char):
        output_argmax = output.permute(1, 0, 2).argmax(dim=-1)
        output_argmax = output_argmax.cpu().numpy()
        output_labels = output_argmax.tolist()
        pred_labels = []

        # 删除ctc_blank
        for label in output_labels:
            pred_label= []
            preNum = label[0]
            for curNum in label[1: ]:
                if preNum == self.ctc_blank:
                    pass
                elif curNum == preNum:
                    pass
                else:
                    char=label2char[preNum]
                    pred_label.append(char)
                preNum = curNum
            pred_labels.append("".join(pred_label))

        return pred_labels