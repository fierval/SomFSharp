namespace FSom

open System
open System.Linq
open System.Collections.Generic
open System.IO

[<AutoOpen>]
module SaverBuilderModule =
    type SaverBuilder (fileName) =
        let buffer = new System.Collections.Generic.List<string>()
        let fileName = fileName
        let buildStringSeq arr =
            let height = Array2D.length1 arr
            seq {
                for i = 0 to height - 1 do 
                    yield String (arr.[i..i, 0..] |> Array2D.map(fun a -> a.ToString())
                        |> Seq.cast<string> |> Seq.fold (fun st e -> st + "\t" + e) String.Empty |> Seq.skip 1 |> Seq.toArray)
            }

        do 
            if String.IsNullOrWhiteSpace fileName then failwith "File name must be specified"
            if (File.Exists fileName) then File.Delete fileName

        member this.Delay f :'a [,] = f()
        member this.Run (x: 'a [,]) = x

        member this.Yield x = 
            Array2D.init 1 1 (fun i j -> x)

        member this.For (seqT, f) =
            seq {
                for s in seqT do
                let x = f s
                yield x
            } |> Seq.last

        [<CustomOperation("add", MaintainsVariableSpace=false)>]
        member this.AddLine (s, line) =
            buffer.Add(line)
            buffer.Add("==================")
            seq{yield ()}
        
        [<CustomOperation("write", MaintainsVariableSpace=false)>]
        member this.Write (s, x) =
            buffer.AddRange(x |> buildStringSeq)
            x

        [<CustomOperation("commit", MaintainsVariableSpace=false)>]
        member this.Commit (s) =
            File.WriteAllLines(fileName, buffer)
            s
    let saver x = SaverBuilder(x)    

    let arr1 = Array2D.init 5 7 (fun i j -> i + j)
    let arr2 = Array2D.init 8 9 (fun i j -> i * j)

    let x = saver @"c:\temp\t1.txt" {
        write arr1
        add "Separator"
        write arr2
        commit
    }
