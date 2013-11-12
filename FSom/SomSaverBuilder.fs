namespace FSom

open System
open System.Linq
open System.Collections.Generic
open System.IO

[<AutoOpen>]
module SomSaverBuilderModule =
    type SomSaverBuilder (fileName) =
        let buffer = new System.Collections.Generic.List<string>()
        let fileName = fileName
        let buildStringSeq (arr : 'a [,]) =
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

        member this.Return (x) = this.Yield x

        member this.Zero () = Array2D.init 0 0 (fun i j -> Unchecked.defaultof<'a>)

        member this.Yield x = 
            File.WriteAllLines(fileName, buffer)
            Array2D.init 1 1 (fun i j -> x)

        member this.For (seqT, f) =
            seq {
                for s in seqT do
                let x = f s
                buffer.AddRange(x |> buildStringSeq)
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
            s

    let som_saver x = SomSaverBuilder(x)    

    let arr1 = Array2D.init 5 7 (fun i j -> i + j)
    let arr2 = Array2D.init 8 9 (fun i j -> i * j)

    let x = som_saver @"c:\temp\t1.txt" {
        write arr1
        add "Separator"
        write arr2
        return "Done"
    }
    

    

