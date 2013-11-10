namespace FSom

open MathNet.Numerics.LinearAlgebra.Double
open System.Threading
open System.Threading.Tasks
open System.Collections.Generic
open System.Linq
open Microsoft.FSharp.Collections
open System

[<AutoOpen>]
module clustering =
   
    type Labels =
    | Unlabeled = -1
    | Watershed = 0

    type Direction =
    | Ascend = 0
    | Descend = 1

    let (|Ascend|Descend|) = function
        |Direction.Ascend -> Ascend
        |Direction.Descend -> Descend
        |_ -> failwith "Impossible direction"
        
    type Point =
    |Point of X : int * Y : int

        member this.xy = 
            match this with
            | Point(X, Y) -> X, Y
        
        member this.x =
            match this with
            |Point(X = x) -> x

        member this.y =
            match this with
            |Point(Y = y) -> y

        static member Copy (point : Point) = Point(point.xy)

        //neighbors of a point on a rectangular grid
        member this.getNeighbors width height =
            let row, col = this.xy
            let neighbors = List<int*int>()
            for i = -1 to 1 do
                for j = -1 to 1 do
                    if i <> 0 && j <> 0 && i + row >= 0 && j + col >= 0 && i + row < height && j + col < width then
                        neighbors.Add(row + i, col + j)
            neighbors

    type AscentDescent(uMatrix : float[,], pMatrix : float [,]) =
        let uMatrix = uMatrix
        let pMatrix = pMatrix
        let width = uMatrix |> Array2D.length2
        let height = uMatrix |> Array2D.length1
        
        [<DefaultValue>] val mutable Immersion : Point[,]

        // ascent/descent parameters: minDistance between points
        // where the process should stop
        let precision = 0.00001

        //max number of iterations after which the process stops
        let maxIterations = (width + height) / 2

        // move around the matrix whether ascending or descending
        let ascendOrDescend (point : Point) (direction : Direction) =
            let mutable stop = false
            let mutable movingPoint = Point.Copy(point)
            let mutable iteration = 0
            let mutable resX = -1
            let mutable resY = -1
            let m =
                match direction with
                | Ascend -> pMatrix
                | Descend -> uMatrix

            while not stop do
                let neighbors = movingPoint.getNeighbors width height
                let minPoint = 
                    neighbors
                    |> 
                    match direction with
                    |Ascend ->
                        Seq.maxBy
                            (fun p ->
                                let x, y = Point(p).xy
                                m.[x,y]
                                )
                    |Descend ->
                        Seq.minBy
                            (fun p ->
                                let x, y = Point(p).xy
                                m.[x,y]
                                )
                
                // index of minimal/maximal point    
                let X, Y = Point(minPoint).xy
                let x, y = movingPoint.xy
                
                let currentMinMax = 
                    match direction with
                    | Ascend ->  if m.[X,Y] >= m.[x,y] then Point(X, Y) else Point(x, y)
                    | Descend -> if m.[X,Y] <= m.[x,y] then Point(X, Y) else Point(x, y)
                
                resX <- currentMinMax.x
                resY <- currentMinMax.y

                if abs (m.[x,y] - m.[X, Y]) < precision then stop <- true
                if iteration >= maxIterations then stop <- true
                iteration <- iteration + 1
            Point(resX, resY)

        // produces the immersion matrix
        member this.Immerse () =
            // descent matrix maps all points to their "minimums"
            let descent = 
                Array2D.init height width
                    (fun i j -> ascendOrDescend (Point(i, j)) Direction.Descend)
            this.Immersion <-
                Array2D.init height width
                    (fun i j -> ascendOrDescend descent.[i,j] Direction.Descend)
                                  
    type Watershed (uMatrix: float [,]) =
        let uMatrix = uMatrix
        let width = uMatrix |> Array2D.length2
        let height = uMatrix |> Array2D.length1

        let labels = Array2D.init height width (fun i j -> int Labels.Unlabeled)
        let minimalpoints = Array2D.zeroCreate height width
        let localmins : (int * int) [] = Array.zeroCreate (height * width) 
        let mutable basins = 0

        /// given a u-matrix retrun a map of its heights
        let getOrderedHeights () =
            let heightMap = Dictionary<float, System.Collections.Generic.List<int*int>>()
            uMatrix 
            |> Array2D.iteri
                (fun i j h -> 
                    if not (heightMap.ContainsKey h) then 
                        heightMap.Add(h, List<int*int>())
                    let lst = heightMap.[h]
                    lst.Add(i,j)
                )
            heightMap.OrderBy(fun kvp -> kvp.Key)

        let processShed (point : Point) =
            let neighbors = point.getNeighbors width height
            
            let availableLabels = HashSet(HashIdentity.Structural)

            let mutable minNeighbor = Unchecked.defaultof<int*int>
            let mutable minNeighborHeight = Double.MaxValue
            let mutable nUnlabeled = 0
            let mutable nBorders = 0

            for neighbor in neighbors do
                let row, col = fst neighbor, snd neighbor
                let marker = labels.[row, col]
                let height = uMatrix.[row, col]
                if marker <> int Labels.Unlabeled && height < minNeighborHeight then
                    minNeighbor <- neighbor
                    minNeighborHeight <- height

                match marker with 
                | m when m = int Labels.Unlabeled -> nUnlabeled <- nUnlabeled + 1
                | n when n = int Labels.Watershed -> nBorders <- nBorders + 1
                | _ -> availableLabels.Add(marker)
            
            let row, col = point.xy
            if availableLabels.Count = 0 then
                if nUnlabeled > 0 then
                    // new basin
                    basins <- basins + 1
                    labels.[row, col] <- basins
                    minimalpoints.[row, col] <- point.xy
                    localmins.[basins] <- point.xy
                else
                    // watershed
                    labels.[row, col] <- int Labels.Watershed
                    minimalpoints.[row, col] <- fst minNeighbor, snd minNeighbor
            elif availableLabels.Count = 1 then
                // belong to an old basin
                let marker = availableLabels.First()
                labels.[row, col] <- marker
                minimalpoints.[row, col] <- fst minNeighbor, snd minNeighbor
            else
                // more than one label, hence - watershed!
                labels.[row, col] <- int Labels.Watershed
                
                
        /// Actually create the watershed
        member this.CreateWatersheds () =
            let heightMap = getOrderedHeights()
            heightMap.Select(
                    fun kvp -> 
                        let points = kvp.Value
                        points |> Seq.iter (fun p -> processShed (Point(p)))
            )
   
