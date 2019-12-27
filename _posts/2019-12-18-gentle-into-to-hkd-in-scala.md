---
layout: post
title:  "[DRAFT] A gentle introduction to higher kinded data in Scala"
date:   2019-12-18 13:00:00 +0100
categories: scala hkd
---
{% include code_blocks_init.html %}

In Scala, we have higher kinded types. Using these, we can abstract over types that themselves take types, like `List`. You've probably seen higher kinded types used in typeclasses like `Functor` and friends, or in tagless final. Their usage often looks like this.
```scala
trait Functor[F[_]] {
  def map[A, B](fa: F[A])(f: A => B): F[B]
}

trait UserService[F[_]] {
  def getUser(id: UserId): F[User]
  
  def updateUser(user: User): F[Unit]
}
```

Today, however, I want to talk about another usage of higher kinded types called higher kinded data (HKD).

## A mountain of boilerplate

Say you're working at some REST endpoints, and need to return some data for the first endpoint, and allow it to be patched by another. Let's use an additional `Option`, to indicate if a JSON property was `undefined`.
```scala
//For this post, I'll use Circe and doobie for Json and SQL

def showProject(projectId: String): IO[Result] = {
  projects.withId(projectId).map(_.fold(NotFound)(a => Ok(a.asJson)))
}

def patchProject(projectId: String, json: Json): Result = {
  val root = json.hcursor
  val settings = root.downField("settings")
  
  val partialProjectResult = (
    withUndefined[String]("name", root),
    withUndefined[String]("description", root),
    withUndefined[List[String]]("keywords", json),
    withUndefined[Option[String]]("issues", json),
    withUndefined[Option[String]]("sources", json),
    withUndefined[Option[String]]("support_channel", json),
  ).mapN(PartialProject.apply)
  
  
  //Imagine what handleDecodeError looks like yourself
  partialProjectResult.fold(handleDecodeError) { partialProject =>
    val sets = Fragments.setOpt(
      partialProject.name.map(name => fr"name = $name"),
      partialProject.description.map(description => fr"description = $description"),
      partialProject.keywords.map(keywords => fr"keywords= $keywords"),
      partialProject.issues.map(issues => fr"issues = $issues"),
      partialProject.sources.map(sources => fr"sources = $sources"),
      partialProject.supportChannel.map(supportChannels => fr"support_channel = $supportChannel"),
    )
  
    val query = sql"UPDATE projects  " ++ sets ++ fr"WHERE project_id = $projectId"
  
    val hasAnyUpdates = partialProject.name.isDefined || 
      partialProject.description.isDefined || partialProject.keywords.isDefined || 
      partialProject.issues.isDefined || partialProject.sources.isDefined || 
      partialProject.supportChannel.isDefined
  
    if (hasAnyUpdates) dbService.run(query.update.run).map(_ => NoContent)
    else IO.pure(BadRequest("No updates specified"))
  }
}

def withUndefined[A: Decoder](
    field: String, cursor: ACursor
): Decoder.AccumulatingResult[Option[A]] = {
  val result = if(cursor.succeeded) Some(cursor.get[A](field)) else None
  result.sequence.toValidatedNel
}

@SnakeCaseDecoder case class Project(
  name: String,
  description: String,
  keywords: List[String],
  issues: Option[String],
  sources: Option[String],
  supportChannel: Option[String]
)

case class PartialProject(
  name: Option[String],
  description: Option[String],
  keywords: Option[List[String]],
  issues: Option[Option[String]],
  sources: Option[Option[String]],
  supportChannel: Option[Option[String]]
)
```

That's a lot of code just for two endpoints. Luckily Circe can help us derive most of it in the get case, but for the patch case, we're not as lucky. Even worse, we have a lot more REST endpoints like this left to implement. What we might look at first is to maybe create a new typeclass for decoding a JSON value that will be used in a patch endpoint. That gets rid of one case of boilerplate, but there's still a lot left. Let's take a second look at the patch method, and replace all cases of boilerplate with `...`. Let's also parameterize all the cases where we refer to something specific.

```scala
def patchThing(table: String, identifierColumn: String)(
    identifier: String, json: Json
): Result = {
  val partialThingResult = (
    ...
  ).mapN(PartialThing.apply)
  
  
  partialThingResult.fold(handleDecodeError) { partialThing =>
    val sets = Fragments.setOpt(
        ...
    )
  
    val query = fr"UPDATE" ++ Fragment.const(table) ++ sets ++ fr"WHERE" ++ 
      Fragment.const(identifierColumn) ++ fr"= $identifier"
  
    val hasAnyUpdates = ...
  
    if (hasAnyUpdates) dbService.run(query.update.run).map(_ => NoContent)
    else IO.pure(BadRequest("No updates specified"))
  }
}
```

Okay, that's not as bad. If we just sprinkle in some shapeless records and loop over it a few times, we should have the method we want, right? Yes, that is right, but it also lands us right into logic programming land. Shapeless is nice for derivation of typeclasses and such, where you can just trust the type signature. It's a bit worse in actual application code. Is there another way to solve this problem that does not include using shapeless at all, and shows more of our intent? Yes, there is.

## Introducing ProjectF
Look back at the boilerplate mountain. We wrote two case classes, with the same fields, just that one had everything wrapped in `Option`. Let's instead define a single case class with a higher kinded type parameter, which indicates the wrapping type.
```scala
case class ProjectF[F[_]](
  name: F[String],
  description: F[String],
  // Note that I don't wrap this in F, as it's content will be wrapped in F instead
  settings: ProjectSettings[F]
)

case class ProjectSettingsF[F[_]](
  keywords: F[List[String]],
  issues: F[Option[String]],
  sources: F[Option[String]],
  supportChannel: F[Option[String]]
)
```

We can now get back `PartialProject` like so.
```scala
type PartialProject = ProjectF[Option]
```

Can we also use this case class for the non-partial case class? Yes, using the `Id` type. The `Id` type just spits back out what we throw at it. Think of it like a type level `Predef.identity`.
```scala
type Id[A] = A
type Project = ProjectF[Id]
```

### Const is a really useful type

There is one more important type and instance we need, `Const`. (When the Scala and Dotty representation of a concept differs substantially, I'll include both.)
{% capture scala-const %}
// Here we're defining a partially applied type. We use it like so 
// Const[String]#λ[Int], or if we're in a place expecting a higher kinded 
// type, just Const[String]#λ
type Const[A] = {
  type λ[B] = A
}
{% endcapture %}

{% capture dotty-const %}
type Const[A] = [B] =>> A
{% endcapture %}

{% include code_blocks_code.html scala=scala-const dotty=dotty-const id="const-type" %}

It is, in some ways, opposite to how `Id` works. While `Id` spits back what we threw at it, `Const` ignores that and instead spits back what it was initially applied with. Here's an example.
{% capture scala-const-usage %}
type Name[A] = Const[String]#λ[A]

type Foo = Name[Int] // Type of Foo is String
type Bar = Name[String] // Type of Bar is String
type Baz = Name[Option[List[Double]]] // Type of Baz is String
type Bin = Name[Nothing] // Type of Bin is String
{% endcapture %}

{% capture dotty-const-usage %}
type Name[A] = Const[String][A]

type Foo = Name[Int] // Type of Foo is String
type Bar = Name[String] // Type of Bar is String
type Baz = Name[Option[List[Double]]] // Type of Baz is String
type Bin = Name[Nothing] // Type of Bin is String
{% endcapture %}

{% include code_blocks_code.html scala=scala-const-usage dotty=dotty-const-usage id="const-type-usage" %}
What's so important about `Const`? It allows us to put any type into `ProjectF` we want, as long as it's the same everywhere. I just showed one such use case. `Const[List[String]]` allows us to put the field names into the structure. We need the list here to also account for parents in nested HKD structures. (More on nested HKD much later)

{% capture scala-projectF-names %}
object ProjectF {
  val names: ProjectF[Const[List[String]]#λ] = ProjectF[Const[List[String]]#λ](
    List("name"),
    List("description"),
    ProjectSettingsF[Const[List[String]]#λ](
      List("settings", "keywords"),
      List("settings", "issues"),
      List("settings", "sources"),
      List("settings", "supportChannel")
    )
  )
}
{% endcapture %}

{% capture dotty-projectF-names %}
object ProjectF with
  val names: ProjectF[Const[String]] = ProjectF(
    List("name"),
    List("description"),
    ProjectSettingsF(
      List("settings", "keywords"),
      List("settings", "issues"),
      List("settings", "sources"),
      List("settings", "supportChannel")
    )
  )
{% endcapture %}

{% include code_blocks_code.html scala=scala-projectF-names dotty=dotty-projectF-names id="projectF-names" %}
This is one of the things it can be nice to have a macro generate, but for now, we'll write it out manually. Anyway, that's pretty nice, just one problem. In many of our cases, we're using `snake_case`. We could just redefine `ProjectF`, but what if we instead made a function that transforms the strings in the structure?

{% capture scala-projectF-names-transform %}
object ProjectF {
  val names: ProjectF[Const[List[String]]#λ] = ProjectF[Const[List[String]]#λ](
    List("name"),
    List("description"),
    ProjectSettingsF[Const[List[String]]#λ](
      List("settings", "keywords"),
      List("settings", "issues"),
      List("settings", "sources"),
      List("settings", "supportChannel")
    )
  )
  
  def transformNames(oldNames: ProjectF[Const[List[String]]#λ])(
      f: String => String
  ): ProjectF[Const[List[String]]#λ] = ProjectF[Const[List[String]]#λ](
    oldNames.name.map(f),
    oldNames.description.map(f),
    ProjectSettingsF[Const[List[String]]#λ](
      oldNames.settings.keywords.map(f),
      oldNames.settings.issues.map(f),
      oldNames.settings.sources.map(f),
      oldNames.settings.supportChannel.map(f)
    )
  )
  
  // Imagine yourself where snakeCaseRename comes from
  val snakeCaseNames: ProjectF[Const[List[String]]#λ] = transformNames(names)(snakeCaseRename)
}
{% endcapture %}

{% capture dotty-projectF-names-transform %}
object ProjectF with
  val names: ProjectF[Const[List[String]]] = ProjectF(
    List("name"),
    List("description"),
    ProjectSettingsF(
      List("settings", "keywords"),
      List("settings", "issues"),
      List("settings", "sources"),
      List("settings", "supportChannel")
    )
  )
  
  def transformNames(oldNames: ProjectF[Const[List[String]]])(
      f: String => String
  ): ProjectF[Const[String]#λ] = ProjectF(
    oldNames.name.map(f),
    oldNames.description.map(f),
    ProjectSettingsF(
      oldNames.settings.keywords.map(f),
      oldNames.settings.issues.map(f),
      oldNames.settings.sources.map(f),
      oldNames.settings.supportChannel.map(f)
    )
  )
  
  // Imagine yourself where snakeCaseRename comes from
  val snakeCaseNames: ProjectF[Const[List[String]]] = transformNames(names)(snakeCaseRename)
{% endcapture %}

{% include code_blocks_code.html scala=scala-projectF-names-transform dotty=dotty-projectF-names-transform id="projectF-names-transform" %}

## Let's implement some typeclasses

Wait... We just applied a function over the entire structure. Can we do this with any type? Isn't that what a functor is? Yes, and `ProjectF[Const]` has one. Let's define it.

{% capture scala-projectF-const-functor %}
object ProjectF {
  ... // All the stuff we defined before
   
  implicit val projectConstFunctor: Functor[λ[A => ProjectF[Const[A]#λ]]] = new Functor[λ[A => ProjectF[Const[A]#λ]]] {
    override def map[A, B](fa: ProjectF[Const[A]#λ])(f: A => B): ProjectF[Const[B]#λ] = ProjectF[Const[B]#λ](
      f(fa.name),
      f(fa.description),
      f(fa.keywords),
      f(fa.issues),
      f(fa.sources),
      f(fa.supportChannel)
    )
  }
}
{% endcapture %}

{% capture dotty-projectF-const-functor %}
object ProjectF with
  ... // All the stuff we defined before
   
  given Functor[[A] =>> ProjectF[Const[A]]]:
    override def [A, B](fa: ProjectF[Const[A]]) map(f: A => B): ProjectF[Const[B]] = ProjectF(
      f(fa.name),
      f(fa.description),
      f(fa.keywords),
      f(fa.issues),
      f(fa.sources),
      f(fa.supportChannel)
    )
{% endcapture %}

{% include code_blocks_code.html scala=scala-projectF-const-functor dotty=dotty-projectF-const-functor id="projectF-const-functor" %}

Okay, so we got some nice abstraction for `Const`. Can we generalize it further, what would that look like? Currently, we have a `map` function that takes in a `ProjectF[Const[A]#λ]`, and returns a `ProjectF[Const[B]#λ]`. What if we could instead define a function that takes a `ProjectF[A]`, and returns a `ProjectF[B]`, where `A` and `B` are higher kinded types? That sounds like a functor on `ProjectF`. Before we define this, we need yet another type. `A => B` just won't be enough anymore.

### Natural transformations
What we need is to somehow be able to pass something like this as a value.
```scala
def headOption[A](xs: List[A]): Option[A] = xs.headOption
```

We can pass `List[Int] => Option[Int]` and `List[String] => Option[String]` as values, but `List => Option` isn't valid. Luckily there is a way to encode what we want. We can define a new type `FunctionK`, and alias it to `~>:`. I throw on a `:` here as I prefer my arrows to associate in the right direction.

{% capture scala-functionK %}
trait FunctionK[A[_], B[_]] {
  def apply[Z](a: A[Z]): B[Z]
}
object FunctionK {
  
  def identity[F[_]]: F ~>: F = λ[F ~>: F](Predef.identity(_))

  def const[F[_], A](a: A): F ~>: Const[A]#λ = new FunctionK[F, Const[A]#λ] {
    override def apply[Z](fz: F[Z]): A = a
  }
}

// Stick this in some package object somewhere
type ~>:[A[_], B[_]] = FunctionK[A, B]
{% endcapture %}

{% capture dotty-functionK %}
// Luckily Dotty already has an encoding for these, 
// so we'll just add a few type aliases 
type FunctionK[A[_], B[_]] = [Z] => A[Z] => B[Z]
type ~>:[A[_], B[_]] = FunctionK[A, B]

object FunctionK with
  def identity[F[_]]: F ~>: F = [Z] => (fz: F[Z]) => fz

  def const[F[_], A](a: A): F ~>: Const[A] = [Z] => (fz: F[Z]) => a
{% endcapture %}

{% include code_blocks_code.html scala=scala-functionK dotty=dotty-functionK id="functionK" %}

We can create and use them like this.
{% capture scala-functionK-usage %}
// Normal usage looks like so. We need to create a new instance of the class in 
// the same way you had for functions in Java before Java 8.
val headOption1: List ~>: Option = new (List ~>: Option) {
  override def apply[Z](fa: List[Z]): Option[Z] = fa.headOption
}

// We can however also use Kind projector in simple cases, but then we loose the 
// ability to refer to the type.
val headOption2: List ~>: Option = λ[List ~>: Option](_.headOption)

val optHead: Option[Int] = headOption(Nil)
{% endcapture %}

{% capture dotty-functionK-usage %}
// No underscore syntax here. You must define both the type, and the 
// parameter with the type applied. 
val headOption: List ~>: Option = [Z] => (a: List[Z]) => a.headOption

val optHead: Option[Int] = headOption(Nil)
{% endcapture %}

{% include code_blocks_code.html scala=scala-functionK-usage dotty=dotty-functionK-usage id="functionK-usage" %}

### FunctorK
We now have almost all the pieces we need. We just need a new functor typeclass which can handle our new types.

{% capture scala-functorK %}
trait FunctorK[F[_[_], _]] {
  def map[A[_], B[_], C](fa: F[A, C])(f: A ~>: B): F[B, C]
  
  def lift[A[_], B[_], C](f: A ~>: B): F[A, *] ~>: F[B, *] =
    λ[F[A, *] ~>: F[B, *]](fa => mapK(fa)(f))
}
{% endcapture %}

{% capture dotty-functorK %}
trait FunctorK with
  def [A[_], B[_], C](fa: F[A, C]) map(f: A ~>: B): F[B, C]
  
  def lift[A[_], B[_], C](f: A ~>: B): F[A, *] ~>: F[B, *] = 
    [Z] => (a: F[A, Z]) => a.map(f)
{% endcapture %}

{% include code_blocks_code.html scala=scala-functorK dotty=dotty-functorK id="functorK" %}

Wait, what's with the `C` type everywhere, why does `F` look like that. That's one of the things I'd rather not go into here right now as it complicates stuff a bit, but as a quick answer, take a look at lift, and tell me how you could get a return type of `F[A] ~>: F[B]` if we didn't have `C` around. For our use case with ProjectF, we can define this to ignore it for the most part.

{% capture scala-functorKC %}
// Stick these in some package object somewhere

// Don't really have a good name for this one
type IgnoreC[F[_[_]]] = {
  type λ[A[_], _] = F[A]
}
type FunctorKC[F[_[_]]] = FunctorK[IgnoreC[F]#λ]
{% endcapture %}

{% capture dotty-functorKC %}
// Don't really have a good name for this one
type IgnoreC[F[_[_]]] = [A[_], _] =>> F[A]
type FunctorKC[F[_[_]]] = FunctorK[IgnoreC[F]]
{% endcapture %}

{% include code_blocks_code.html scala=scala-functorKC dotty=dotty-functorKC id="functorKC" %}

### ApplyK and ApplicativeK
Great, we have our `FunctorK` typeclass. Let's get a few more. Next up is `ApplyK`, but before we create that one, we need to encode tuples.

{% capture scala-tupleK %}
type Tuple2K[F[_], G[_]] = { 
  type λ[A] = (F[A], G[A]) 
}
{% endcapture %}

{% capture dotty-tupleK %}
type Tuple2K[A[_], B[_]] = [Z] =>> (A[Z], B[Z])
{% endcapture %}

{% include code_blocks_code.html scala=scala-tupleK dotty=dotty-tupleK id="tupleK" %}

This type lets us have the same type, but with different wrappers. For example, using `ProjectF[Tuple2K[List, Option]#λ]` gives us something like this.
```scala
case class TupledProject(
  name: (List[String], Option[String]),
  description: (List[String], Option[String]),
  keywords: (List[List[String]], Option[List[String]]),
  issues: (List[Option[String]], Option[Option[String]]),
  sources: (List[Option[String]], Option[Option[String]]),
  supportChannel: (List[Option[String]], Option[Option[String]])
)
```
With that out of the way, let's define `ApplyK`. This is probably the most useful typeclass for HKD I think.

{% capture scala-applyK %}
trait ApplyK[F[_[_], _]] extends FunctorK[F] {
  def apK[A[_], B[_], C](ff: F[λ[D => A[D] => B[D]], C])(fa: F[A, C]): F[B, C] =
    map2K(ff, fa)(λ[Tuple2K[λ[D => A[D] => B[D]], A]#λ ~>: B](t => t._1(t._2)))

  def tupledK[A[_], B[_], C](fa: F[A, C], fb: F[B, C]): F[Tuple2K[A, B]#λ, C] =
    map2K(fa, fb)(FunctionK.identity)

  def map2K[A[_], B[_], Z[_], C](fa: F[A, C], fb: F[B, C])(f: Tuple2K[A, B]#λ ~>: Z): F[Z, C]
}

// Stick this in some package object
type ApplyKC[F[_[_]]] = ApplyK[IgnoreC[F]]
{% endcapture %}

{% capture dotty-applyK %}
trait ApplyK[F[_[_], _]] extends FunctorK[F]
  def [A[_], B[_], C](ff: F[[D] =>> A[D] => B[D], C]) ap(fa: F[A, C]): F[B, C] =
    ff.map2K(fa)([Z] => (t: Tuple2K[[D] =>> A[D] => B[D], A][Z]) => t._1(t._2))

  def [A[_], B[_], Z[_], C](fa: F[A, C]) map2K(fb: F[B, C])(f: Tuple2K[A, B] ~>: Z): F[Z, C]

  def [A[_], B[_], C](fa: F[A, C]) tupledK(fb: F[B, C]): F[Tuple2K[A, B], C] = 
    fa.map2K(fb)(FunctionK.identity[Tuple2K[A, B]])

type ApplyKC[F[_[_]]] = ApplyK[IgnoreC[F]]
{% endcapture %}

{% include code_blocks_code.html scala=scala-applyK dotty=dotty-applyK id="applyK" %}

We can also define the least useful typeclass, `ApplicativeK`. Why is it so useless? Because unlike with the normal applicative, there aren't many cases where we want to construct a new instance of our type. In fact, doing so is hard because we need to be able to construct `A[Z]`, for all types `Z`. Either you can use `Const`, `A[Nothing]` where `A` is covariant, or `A[Any]`, where `A` is contravariant.

{% capture scala-applicativeK %}
trait ApplicativeK[F[_[_], _]] extends ApplyK[F] {

  def pureK[A[_], C](a: Const[Unit]#λ ~>: A): F[A, C]

  def unitK[C]: F[Const[Unit]#λ, C] = pureK(FunctionK.identity)

  override def mapK[A[_], B[_], C](fa: F[A, C])(f: A ~>: B): F[B, C] =
    apK(pureK[λ[D => A[D] => B[D]], C](λ[Const[Unit]#λ ~>: λ[D => A[D] => B[D]]](_ => f.apply)))(fa)
}

// Stick this in some package object
type ApplicativeKC[F[_[_]]] = ApplicativeK[IgnoreC[F]]
{% endcapture %}

{% capture dotty-applicativeK %}
trait ApplicativeK[F[_[_], _]] extends ApplyK[F]
  def [A[_], C](a: Const[Unit] ~>: A) pure: F[A, C]

  def unitK[C]: F[Const[Unit], C] = ValueK.const(()).pure

  override def [A[_], B[_], C](fa: F[A, C]) mapK(f: A ~>: B): F[B, C] =
    ([Z] => () => f[Z]).pure[[D] =>> A[D] => B[D], C].ap(fa)

type ApplicativeKC[F[_[_]]] = ApplicativeK[IgnoreC[F]]
{% endcapture %}

{% include code_blocks_code.html scala=scala-applicativeK dotty=dotty-applicativeK id="applicativeK" %}

## First step away from the boilerplate
While we're still missing a few pieces that we're going to need, we can begin to look at how we can use these typeclasses to get rid of the boilerplate. Let's focus on this piece of code.
```scala
val root = json.hcursor
val settings = root.downField("settings")

val partialProjectResult = (
  withUndefined[String]("name", root),
  withUndefined[String]("description", root),
  withUndefined[List[String]]("keywords", json),
  withUndefined[Option[String]]("issues", json),
  withUndefined[Option[String]]("sources", json),
  withUndefined[Option[String]]("support_channel", json),
).mapN(PartialProject.apply)
```

What do we need here? We need the names which we already have and the decoders. Let's make a new instance of `ProjectF` filled with decoders.
```scala
val projectDecoders: ProjectF[Decoder] = ProjectF[Decoder](
  Decoder[String],
  Decoder[String],
  ProjectSettingsF[Decoder](
    Decoder[List[String]],
    Decoder[Option[String]],
    Decoder[Option[String]],
    Decoder[Option[String]],
  )
)
```

We also need the cursor to use. Take all that, blend it together, and we get a method to decode an HKD type from a patch payload.

{% capture scala-patch-decode-wrong %}
def patchDecode[F[_[_], _], C](names: F[Const[List[String]]#λ, C], decoders: F[Decoder, C], cursor: ACursor)(
    implicit F: ApplyK[F]
): F[λ[A => Decoder.AccumulatingResult[Option[A]]], C] =
  F.map2K(names, decoders)(
    new (Tuple2K[Const[List[String]]#λ, Decoder]#λ ~>: λ[A => Decoder.AccumulatingResult[Option[A]]]) {
      override def apply[Z](t: (List[String], Decoder[Z])): Decoder.AccumulatingResult[Option[Z]] = {
        val names   = t._1
        val decoder = t._2

        val cursorWithNames = names.foldLeft(cursor)(_.downField(_))

        val result = if (cursorWithNames.succeeded) Some(cursorWithNames.as[Z](decoder)) else None
        result.sequence.toValidatedNel
      }
    }
  )
{% endcapture %}

{% capture dotty-patch-decode-wrong %}
def patchDecode[F[_[_], _], C](names: F[Const[List[String]], C], decoders: F[Decoder, C], cursor: ACursor)(
    given ApplyK[F]
): F[[A] =>> Decoder.AccumulatingResult[Option[A]], C] =
  names.map2K(decoders) { [Z] => (t: (List[String], Decoder[Z])) =>
    val names   = fa._1
    val decoder = fa._2
    
    val cursorWithNames = names.foldLeft(cursor)(_.downField(_))

    val result = if (cursorWithNames.succeeded) Some(cursorWithNames.as[Z](decoder)) else None
    result.sequence.toValidatedNel
  }
{% endcapture %}

{% include code_blocks_code.html scala=scala-patch-decode-wrong dotty=dotty-patch-decode-wrong id="patch-decode-wrong" %}

Wonderful, this is what we want, right? Almost. Just one problem left. Using this with `ProjectF`, it gives us a `ProjectF[λ[A => Decoder.AccumulatingResult[Option[A]]]]`, but what we want is a `Decoder.AccumulatingResult[ProjectF[Option]]`. That sounds like a call to `sequence`. Guess we'll need `TraverseK` too. Before we go off and do `TraverseK` too, let's look at some of the other boilerplate.

```scala
val sets = Fragments.setOpt(
  partialProject.name.map(name => fr"name = $name"),
  partialProject.description.map(description => fr"description = $description"),
  partialProject.keywords.map(keywords => fr"keywords= $keywords"),
  partialProject.issues.map(issues => fr"issues = $issues"),
  partialProject.sources.map(sources => fr"sources = $sources"),
  partialProject.supportChannel.map(supportChannels => fr"support_channel = $supportChannel"),
)

val hasAnyUpdates = partialProject.name.isDefined || 
  partialProject.description.isDefined || partialProject.keywords.isDefined || 
  partialProject.issues.isDefined || partialProject.sources.isDefined || 
  partialProject.supportChannel.isDefined
```

`hasAnyUpdates` is probably the easiest one here, and the only one we could technically solve (if we erased some types) right now. Say that we somehow could convert any `ProjectF[A]` into a `List[A[_]]`. In that case, the problem becomes easy. We just fold over the list. Can we get rid of the list, and just fold over the `ProjectF` directly? If we had `FoldableK` we could.

What about `sets`? `Fragments.setOpt` takes a vararg `Option[Fragment]`, so we probably need `FoldableK` here too, but before that, how do we get the fragments? We probably want our `ProjectF` to store functions from the used type to `Fragment`. Something like `ProjectF[λ[A => (A => Fragment)]]` (I've placed parenthesis around the type to make it easier to read). Once we have the `Option[A]`, we can then map it with the function `A => Fragment`, to get a `Const[Fragment]#λ[A]`. Only one problem in that plan, doobie resists slightly against dealing with HKD, mostly when dealing with nullable columns. We also can't use the interpolator to make our lives easy.

First, we need a type to translate between doobie's handling of `Option` and our handling.

```scala
case class ElemCreator[A](mkElem: A => Param.Elem)
object ElemCreator {
  def arg[A](implicit put: Put[A]): ElemCreator[A]         = ElemCreator(Param.Elem.Arg(_, put))
  def opt[A](implicit put: Put[A]): ElemCreator[Option[A]] = ElemCreator(Param.Elem.Opt(_, put))
}
```

{% capture scala-doobie-creator %}
def createUpdater[F[_[_], _], C](
    names: F[Const[List[String]]#λ, C],
    elemCreators: F[ElemCreator, C]
)(implicit F: ApplyK[F]): F[λ[A => (A => Fragment)], C] =
  F.map2K(names, elemCreators)(new (Tuple2K[Const[List[String]]#λ, ElemCreator]#λ ~>: λ[A => (A => Fragment)]) {
    override def apply[Z](t: (List[String], ElemCreator[Z])): Z => Fragment = {
      val names   = t._1
      val creator = t._2

      (value: Z) => Fragment.const(names.last) ++ Fragment("= ?", List(creator.mkElem(value)))
    }
  })
{% endcapture %}

{% capture dotty-doobie-creator %}
def createUpdater[F[_[_], _], C](
    names: F[Const[List[String]], C],
    elemCreators: F[ElemCreator, C]
)(given ApplyK[F]): F[[A] =>> (A => Fragment), C] =
  names.map2K(elemCreators) { [Z] => (t: (List[String], ElemCreator[Z])) =>
    val names   = t._1
    val creator = t._2

    (value: Z) => Fragment.const(names.last) ++ Fragment("= ?", List(creator.mkElem(value)))
  }
{% endcapture %}

{% include code_blocks_code.html scala=scala-doobie-creator dotty=dotty-doobie-creator id="doobie-creator" %}

Testing